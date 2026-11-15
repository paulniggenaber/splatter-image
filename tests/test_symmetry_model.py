"""
Checks the symmetry model end to end on a CPU (no renderer needed):
  - the released weights load into the symmetry model, only the new heads are missing
  - with the switch forced to 0 ... the first N Gaussians are exactly the baseline's
  - output shapes, plane loss, gradients reach the new heads
  - the true plane from the dataset passes through the object centre and the
    training cameras of a car are spread on both sides of it

Run from the repo root (needs the data and the checkpoint):
    export SHAPENET_DATASET_ROOT=$HOME/data/SRN
    python -m tests.test_symmetry_model $HOME/ckpt/cars
"""
import os
import sys

import torch
from hydra import compose, initialize

from scene.gaussian_predictor import GaussianSplatPredictor
from scene.symmetry import plane_loss, signed_distance, split_plane


def load_cfg(symmetry):
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="default_config",
                       overrides=["+dataset=cars", "+experiment=lpips_100k",
                                  "model.symmetry.enabled={}".format(str(symmetry).lower())])


def main(ckpt_dir):
    torch.manual_seed(0)
    ckpt = torch.load(os.path.join(ckpt_dir, "model_latest.pth"), map_location="cpu")

    cfg_base = load_cfg(False)
    cfg_sym = load_cfg(True)

    base = GaussianSplatPredictor(cfg_base)
    base.load_state_dict(ckpt["model_state_dict"])            # strict: must match exactly
    sym = GaussianSplatPredictor(cfg_sym)
    result = sym.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert len(result.unexpected_keys) == 0, result.unexpected_keys
    assert all(("sym_head" in k) or ("plane_head" in k) for k in result.missing_keys), result.missing_keys
    print("1 ok  released weights load; missing keys are only the new heads:", result.missing_keys)
    base.eval(); sym.eval()

    from datasets.dataset_factory import get_dataset
    cfg_sym.data.subset = 2
    dataset = get_dataset(cfg_sym, "train")
    sample = dataset[0]
    data = {k: v.unsqueeze(0) for k, v in sample.items()}
    B = 1
    N = cfg_sym.data.training_resolution ** 2

    # --- dataset: the true plane -------------------------------------------------
    plane = data["sym_plane"]                                  # (1, 4)
    n, d = split_plane(plane)
    assert abs(plane[0, :3].norm().item() - 1.0) < 1e-4
    # the input camera is the origin of the frame; its view-to-world must be identity
    assert torch.allclose(data["view_to_world_transforms"][0, 0], torch.eye(4), atol=1e-4)
    # the object centre (dataset origin) in the input-camera frame = where the camera
    # at the dataset origin would be ... we get it from the first camera: its centre in
    # dataset coords is c0, so the dataset origin in camera coords is on the optical
    # axis at the camera distance. It must lie on the plane.
    cam_dist = 1.3
    centre = torch.tensor([[[0.0, 0.0, cam_dist]]])
    dist_centre = signed_distance(centre, plane).abs().item()
    print("   distance of the object centre from the true plane: {:.4f}".format(dist_centre))
    assert dist_centre < 0.05, "object centre should lie on the mirror plane"
    sides = signed_distance(data["camera_centers"], plane).flatten()
    print("   camera sides of the 5 views in this sample:", [round(s, 2) for s in sides.tolist()])
    print("2 ok  true plane: unit normal, passes through the object centre")

    # --- model -------------------------------------------------------------------
    inp = data["gt_images"][:, :1]
    v2w = data["view_to_world_transforms"][:, :1]
    quat = data["source_cv2wT_quat"][:, :1]
    with torch.no_grad():
        out_base = base(inp, v2w, quat, None)
        out_off = sym(inp, v2w, quat, None, sym_plane_override=plane, sym_switch_override=0.0)
        out_on = sym(inp, v2w, quat, None, sym_plane_override=plane, sym_switch_override=1.0)
        out_free = sym(inp, v2w, quat, None)

    for k in ["xyz", "rotation", "scaling", "opacity", "features_dc", "features_rest"]:
        assert out_on[k].shape[1] == 2 * N, (k, out_on[k].shape)
        assert torch.allclose(out_off[k][:, :N], out_base[k], atol=1e-5), k
    assert out_off["opacity"][:, N:].abs().max().item() == 0.0
    assert out_free["sym_plane"].shape == (B, 4) and out_free["sym_switch"].shape == (B, N, 1)
    print("3 ok  2N Gaussians; first N identical to the released model; switch 0 = twins invisible")

    switch_start = out_free["sym_switch"].mean().item()
    print("   switch at the start of training: {:.3f} (expected sigmoid(-2) = 0.119)".format(switch_start))
    assert abs(switch_start - 0.1192) < 1e-3

    # twins really are mirror images: same distance, other side
    d_orig = signed_distance(out_on["xyz"][:, :N], plane)
    d_twin = signed_distance(out_on["xyz"][:, N:], plane)
    assert torch.allclose(d_orig, -d_twin, atol=1e-4)
    op = out_base["opacity"][0, :, 0]
    vis = op > 0.5
    near = (d_orig[0, :, 0] * (-d[0, 0])) > 0
    print("   visible Gaussians (opacity > 0.5): {} of {}; on the camera's side of the plane: {:.0f}%".format(
        int(vis.sum()), N, 100.0 * (vis & near).sum().item() / max(int(vis.sum()), 1)))
    print("4 ok  twins sit at the mirrored positions")

    # --- gradients reach the new heads --------------------------------------------
    sym.train()
    out = sym(inp, v2w, quat, None)
    loss = plane_loss(out["sym_plane"], plane) + out["opacity"][:, N:].mean() + out["xyz"][:, N:].mean()
    loss.backward()
    net = sym.network_with_offset
    assert net.sym_head.bias.grad is not None and net.sym_head.bias.grad.abs().sum() > 0
    assert net.plane_head[0].weight.grad.abs().sum() > 0
    print("5 ok  gradients reach the switch head and the plane head")
    print("ALL MODEL TESTS PASSED")


if __name__ == "__main__":
    main(sys.argv[1])
