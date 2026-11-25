# entirely new code file from Paul
# ======================================================================
"""
Symmetry extension for Splatter Image.

Idea: the model predicts the object's mirror plane. Every Gaussian gets a
mirrored twin on the other side of that plane. A per-Gaussian switch in (0, 1)
scales the twin's opacity, so the model can turn mirroring off where the
object is not symmetric.

A plane is 4 numbers (n_x, n_y, n_z, d): all points p with n . p = d.
Everything lives in the frame of the input camera, because that is the frame
in which GaussianSplatPredictor outputs its Gaussians for one input view.

Nothing in this file needs a GPU, so it can be unit-tested on a CPU
(see tests/test_symmetry.py).
"""
import torch
import torch.nn.functional as F

from utils.general_utils import quaternion_raw_multiply


def split_plane(plane):
    """plane (B, 4) -> unit normal n (B, 3), offset d (B, 1)."""
    n = F.normalize(plane[..., :3], dim=-1)
    d = plane[..., 3:]
    return n, d


def signed_distance(xyz, plane):
    """xyz (B, N, 3), plane (B, 4) -> signed distance to the plane (B, N, 1)."""
    n, d = split_plane(plane)
    return (xyz * n.unsqueeze(1)).sum(-1, keepdim=True) - d.unsqueeze(1)


def reflection_matrix(n):
    """Unit normal n (B, 3) -> Householder reflection H = I - 2 n n^T, (B, 3, 3)."""
    eye = torch.eye(3, device=n.device, dtype=n.dtype).unsqueeze(0)
    return eye - 2.0 * n.unsqueeze(2) * n.unsqueeze(1)


def reflect_points(xyz, plane):
    """Mirror points xyz (B, N, 3) across the plane (B, 4)."""
    n, _ = split_plane(plane)
    return xyz - 2.0 * signed_distance(xyz, plane) * n.unsqueeze(1)


def reflect_quaternions(q, n):
    """
    Mirror the orientation of Gaussians.

    q: (B, N, 4) quaternions, real part first. n: (B, 3) unit plane normal.

    A mirrored rotation matrix H R is not a rotation (determinant -1). But a
    Gaussian only cares about its covariance R S^2 R^T, which does not change
    if one axis of R is flipped. So we use R' = H R F with F = diag(-1, 1, 1),
    which IS a rotation and gives exactly the mirrored covariance H Sigma H^T.
    Because H = -Rot(n, 180 deg) and F = -Rot(x, 180 deg), this is
        R' = Rot(n, 180) R Rot(x, 180)   <->   q' = (0, n) * q * (0, 1, 0, 0)
    (checked numerically against the matrix version, error ~1e-15).
    """
    q_n = torch.cat([torch.zeros_like(n[..., :1]), n], dim=-1)      # (B, 4)
    q_n = q_n.unsqueeze(1).expand_as(q)
    q_x = q.new_tensor([0.0, 1.0, 0.0, 0.0]).expand_as(q)
    return quaternion_raw_multiply(quaternion_raw_multiply(q_n, q), q_x)


def mirror_gaussians(g, plane, switch, transform_SHs, twin_dc_delta=None):
    """
    Append a mirrored twin for every Gaussian.

    g: dict with xyz (B,N,3), rotation (B,N,4), scaling (B,N,3),
       opacity (B,N,1), features_dc (B,N,1,3), features_rest (B,N,3,3)
    plane: (B, 4) mirror plane in the same frame as g["xyz"]
    switch: (B, N, 1) in [0, 1], multiplies the twin's opacity
    transform_SHs: GaussianSplatPredictor.transform_SHs. It applies a 3x3
       matrix to the degree-1 colour coefficients. They behave like a vector,
       so the reflection matrix H can be used in place of a rotation.
    twin_dc_delta: optional (B, N, 1, 3), added to the twin's base colour.
       The shape of an object is symmetric, the lighting is not: in SRN the
       light comes from one side, so the hidden side is brighter or darker
       than the visible one. This lets the model correct the twin's colour.

    Returns a dict with the same keys and 2N Gaussians: originals, then twins.
    """
    n, _ = split_plane(plane)

    twin = {
        "xyz": reflect_points(g["xyz"], plane),
        "rotation": reflect_quaternions(g["rotation"], n),
        "scaling": g["scaling"],
        "opacity": g["opacity"] * switch,
        "features_dc": g["features_dc"] if twin_dc_delta is None else g["features_dc"] + twin_dc_delta,
    }
    if g["features_rest"].shape[2] > 0:
        twin["features_rest"] = transform_SHs(g["features_rest"], reflection_matrix(n))
    else:
        twin["features_rest"] = g["features_rest"]

    return {k: torch.cat([g[k], twin[k]], dim=1) for k in twin}


def near_side_mask(xyz, plane):
    """
    (B, N, 1) float mask: 1 for Gaussians on the same side of the plane as the
    input camera. The input camera sits at the origin of the frame, whose
    signed distance to the plane is -d.
    """
    _, d = split_plane(plane)
    return ((signed_distance(xyz, plane) * (-d).unsqueeze(1)) > 0).float()


def symmetrize_posthoc(g, plane, mode, transform_SHs):
    """
    Training-free symmetrisation of an existing reconstruction. Used for the
    go/no-go check and for test B, with the TRUE plane and the released model.

    mode "twin_all":     every Gaussian gets a twin (switch = 1 everywhere).
    mode "replace_far":  keep only the Gaussians on the camera's side of the
                         plane and build the other half purely by mirroring.
    """
    g = {k: g[k] for k in ["xyz", "rotation", "scaling", "opacity", "features_dc", "features_rest"]}
    switch = torch.ones_like(g["opacity"])
    if mode == "twin_all":
        pass
    elif mode == "replace_far":
        g["opacity"] = g["opacity"] * near_side_mask(g["xyz"], plane)
    else:
        raise ValueError("Unknown symmetrisation mode {}".format(mode))
    return mirror_gaussians(g, plane, switch, transform_SHs)


def plane_loss(pred, gt):
    """
    pred, gt: (B, 4). (n, d) and (-n, -d) describe the same plane, so the loss
    does not care about the sign: 1 - |cos(angle)| plus L1 on the offset.
    """
    n_p, d_p = split_plane(pred)
    n_t, d_t = split_plane(gt)
    cos = (n_p * n_t).sum(-1, keepdim=True)
    sign = torch.where(cos >= 0, torch.ones_like(cos), -torch.ones_like(cos)).detach()
    return (1.0 - cos.abs()).mean() + (d_p * sign - d_t).abs().mean()


@torch.no_grad()
def plane_errors(pred, gt):
    """Mean angle error in degrees and mean offset error, for logging."""
    n_p, d_p = split_plane(pred)
    n_t, d_t = split_plane(gt)
    cos = (n_p * n_t).sum(-1, keepdim=True)
    sign = torch.where(cos >= 0, torch.ones_like(cos), -torch.ones_like(cos))
    angle = torch.rad2deg(torch.acos(cos.abs().clamp(max=1.0))).mean().item()
    offset = (d_p * sign - d_t).abs().mean().item()
    return angle, offset


def is_hidden_side_camera(camera_center, plane):
    """
    camera_center (3,), plane (4,) -> True if the camera is on the other side
    of the mirror plane than the input camera (which sits at the origin).
    """
    n, d = split_plane(plane.unsqueeze(0))
    s = (n[0] * camera_center).sum() - d[0, 0]
    return bool((s * d[0, 0]) > 0)
# ======================================================================
