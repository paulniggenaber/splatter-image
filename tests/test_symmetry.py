"""
Test A of the symmetry plan: the mirror maths, on a CPU.

Run from the repo root:
    python -m tests.test_symmetry
"""
import torch

from scene.symmetry import (mirror_gaussians, plane_loss, plane_errors, reflect_points,
                            reflect_quaternions, reflection_matrix, split_plane,
                            near_side_mask, symmetrize_posthoc, is_hidden_side_camera)


def quat_to_matrix(q):
    """Same formula as utils.general_utils.build_rotation, but device-agnostic."""
    q = q / q.norm(dim=-1, keepdim=True)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], dim=-1
    ).reshape(*q.shape[:-1], 3, 3)


def covariance(q, s):
    R = quat_to_matrix(q)
    return R @ torch.diag_embed(s ** 2) @ R.transpose(-1, -2)


def sh_transform_like_the_model(shs, M):
    """Copy of GaussianSplatPredictor.transform_SHs without the class around it."""
    v_to_sh = torch.tensor([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=shs.dtype)
    sh_to_v = v_to_sh.T
    B, N = shs.shape[:2]
    x = shs.permute(0, 1, 3, 2).reshape(B, N * 3, 3)          # b (n rgb) sh_num
    T = sh_to_v.expand(B, 3, 3) @ M[:, :3, :3] @ v_to_sh.expand(B, 3, 3)
    x = x @ T
    return x.reshape(B, N, 3, 3).permute(0, 1, 3, 2)            # b n sh_num rgb


def sh_colour(dc, rest, view_dir):
    """Degree-1 colour as the renderer computes it (up to constants): dc + rest . basis(dir)."""
    x, y, z = view_dir.unbind(-1)
    basis = torch.stack([-y, z, -x], dim=-1)                    # (..., 3) = sh_num
    return dc[..., 0, :] + (rest * basis[..., :, None]).sum(-2)


def random_gaussians(B=2, N=50, dtype=torch.float64):
    torch.manual_seed(0)
    return {
        "xyz": torch.randn(B, N, 3, dtype=dtype),
        "rotation": torch.nn.functional.normalize(torch.randn(B, N, 4, dtype=dtype), dim=-1),
        "scaling": torch.rand(B, N, 3, dtype=dtype) + 0.1,
        "opacity": torch.rand(B, N, 1, dtype=dtype),
        "features_dc": torch.randn(B, N, 1, 3, dtype=dtype),
        "features_rest": torch.randn(B, N, 3, 3, dtype=dtype),
    }


def main():
    dtype = torch.float64
    B, N = 2, 50
    g = random_gaussians(B, N, dtype)
    plane = torch.randn(B, 4, dtype=dtype)
    plane[:, 3] *= 0.5
    n, d = split_plane(plane)
    H = reflection_matrix(n)

    # 1. mirrored points lie at the same distance on the other side, twice = identity
    xyz_m = reflect_points(g["xyz"], plane)
    dist = (g["xyz"] * n[:, None]).sum(-1) - d
    dist_m = (xyz_m * n[:, None]).sum(-1) - d
    assert torch.allclose(dist, -dist_m, atol=1e-10)
    assert torch.allclose(reflect_points(xyz_m, plane), g["xyz"], atol=1e-10)
    print("1 ok  points: mirrored distance flips sign, mirroring twice gives the original")

    # 2. mirrored quaternion is a unit quaternion and gives exactly H Sigma H^T
    q_m = reflect_quaternions(g["rotation"], n)
    assert torch.allclose(q_m.norm(dim=-1), torch.ones(B, N, dtype=dtype), atol=1e-10)
    cov = covariance(g["rotation"], g["scaling"])
    cov_expected = H[:, None] @ cov @ H[:, None].transpose(-1, -2)
    cov_m = covariance(q_m, g["scaling"])
    assert torch.allclose(cov_m, cov_expected, atol=1e-9)
    det = torch.linalg.det(quat_to_matrix(q_m))
    assert torch.allclose(det, torch.ones_like(det), atol=1e-9)
    print("2 ok  orientation: twin covariance == mirrored covariance, twin rotation is proper")

    # 3. colour: the twin seen from direction v looks like the original seen from the
    #    mirrored direction H v
    out = mirror_gaussians(g, plane, torch.ones(B, N, 1, dtype=dtype), sh_transform_like_the_model)
    rest_m = out["features_rest"][:, N:]
    v = torch.nn.functional.normalize(torch.randn(B, N, 3, dtype=dtype), dim=-1)
    v_m = (v[:, :, None, :] @ H[:, None]).squeeze(2)            # H is symmetric
    c_twin = sh_colour(g["features_dc"], rest_m, v)
    c_orig = sh_colour(g["features_dc"], g["features_rest"], v_m)
    assert torch.allclose(c_twin, c_orig, atol=1e-9)
    print("3 ok  colour: twin from direction v == original from the mirrored direction")

    # 4. shapes and bookkeeping of mirror_gaussians
    for k, v_ in out.items():
        assert v_.shape[1] == 2 * N, k
        assert torch.equal(v_[:, :N], g[k]), k                  # originals untouched
    switch = torch.rand(B, N, 1, dtype=dtype)
    out_s = mirror_gaussians(g, plane, switch, sh_transform_like_the_model)
    assert torch.allclose(out_s["opacity"][:, N:], g["opacity"] * switch)
    print("4 ok  bookkeeping: N -> 2N, originals unchanged, twin opacity = opacity * switch")

    # 5. mirroring the twins again lands on the originals (whole dict, covariance-wise)
    twins = {k: v_[:, N:] for k, v_ in out.items()}
    back = mirror_gaussians(twins, plane, torch.ones(B, N, 1, dtype=dtype), sh_transform_like_the_model)
    assert torch.allclose(back["xyz"][:, N:], g["xyz"], atol=1e-9)
    assert torch.allclose(covariance(back["rotation"][:, N:], g["scaling"]), cov, atol=1e-9)
    assert torch.allclose(back["features_rest"][:, N:], g["features_rest"], atol=1e-9)
    print("5 ok  mirror twice == original (position, covariance, colour)")

    # 6. plane loss: zero for the same plane with flipped sign, positive otherwise
    assert plane_loss(plane, plane).item() < 1e-9
    assert plane_loss(-plane, plane).item() < 1e-9
    assert plane_loss(torch.randn(B, 4, dtype=dtype), plane).item() > 1e-3
    angle, offset = plane_errors(-plane, plane)
    assert angle < 1e-4 and offset < 1e-9
    print("6 ok  plane loss ignores the sign of (n, d)")

    # 7. gradients reach the plane and the switch
    plane_p = plane.clone().requires_grad_(True)
    switch_p = switch.clone().requires_grad_(True)
    o = mirror_gaussians(g, plane_p, switch_p, sh_transform_like_the_model)
    (o["xyz"].sum() + o["opacity"].sum() + o["rotation"].sum() + o["features_rest"].sum()).backward()
    assert plane_p.grad.abs().sum() > 0 and switch_p.grad.abs().sum() > 0
    print("7 ok  gradients flow to the plane and to the switch")

    # 8. near-side mask and hidden-side camera test agree with a hand-made case:
    #    plane x = 0.5 (n = +x, d = 0.5). The input camera is at the origin (x < 0.5).
    p = torch.tensor([[1.0, 0.0, 0.0, 0.5]], dtype=dtype)
    pts = torch.tensor([[[0.0, 0, 0], [0.4, 1, 1], [0.6, 0, 0], [2.0, -1, 3]]], dtype=dtype)
    assert near_side_mask(pts, p).flatten().tolist() == [1.0, 1.0, 0.0, 0.0]
    assert not is_hidden_side_camera(torch.tensor([0.0, 0, 0], dtype=dtype), p[0])
    assert is_hidden_side_camera(torch.tensor([1.0, 0, 0], dtype=dtype), p[0])
    # same plane written with the opposite sign must give the same answers
    assert near_side_mask(pts, -p).flatten().tolist() == [1.0, 1.0, 0.0, 0.0]
    assert is_hidden_side_camera(torch.tensor([1.0, 0, 0], dtype=dtype), -p[0])
    g1 = {k: v_[:1, :4] for k, v_ in g.items()}
    g1["xyz"] = pts
    rep = symmetrize_posthoc(g1, p, "replace_far", sh_transform_like_the_model)
    assert rep["opacity"][0, 2:4].abs().sum() == 0 and rep["opacity"][0, 6:8].abs().sum() == 0
    assert torch.allclose(rep["opacity"][0, 4:6], g1["opacity"][0, 0:2])
    print("8 ok  near/hidden side logic, independent of the plane's sign")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
