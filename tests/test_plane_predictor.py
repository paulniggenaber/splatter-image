"""
Test of the stand-alone plane predictor (CPU, a few seconds).
Run from the repo root:  python -m tests.test_plane_predictor
"""
import torch
from hydra import compose, initialize

from scene.plane_predictor import (PlanePredictor, matrix_loss, matrix_target,
                                   normal_from_raw, plane_from_normal, save_predictor)
from scene.symmetry import plane_errors


def main():
    torch.manual_seed(0)

    # 1. n n^T -> n round trip, for any sign
    n = torch.nn.functional.normalize(torch.randn(200, 3), dim=-1)
    for s in (1.0, -1.0):
        n_back = normal_from_raw(matrix_target(s * n))
        cos = (n_back * n).sum(-1).abs()
        assert (cos > 1 - 1e-5).all()
    assert (normal_from_raw(matrix_target(n))[:, 2] >= 0).all()
    print("1 ok  matrix -> normal round trip, sign-free, n_z >= 0")

    # 2. d = r * n_z, plane has unit normal
    plane = plane_from_normal(n, 1.3)
    assert torch.allclose(plane[:, 3], 1.3 * n[:, 2])
    angle, off = plane_errors(plane, torch.cat([n, 1.3 * n[:, 2:3]], dim=1))
    assert angle < 0.1 and off < 1e-6
    print("2 ok  plane = (n, 1.3 n_z), zero error against itself")

    # 3. the model can fit one image -> one plane (loss goes down, angle goes down)
    for arch in ["small", "resnet18"]:
        m = PlanePredictor(arch=arch, pretrained=False)
        img = torch.rand(4, 3, 128, 128)
        nt = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        first = matrix_loss(m.raw(img), nt).item()
        for _ in range(60):
            loss = matrix_loss(m.raw(img), nt)
            opt.zero_grad(); loss.backward(); opt.step()
        pl = m(img)
        assert loss.item() < first * 0.3, (arch, first, loss.item())
        ang, _ = plane_errors(pl, plane_from_normal(nt, 1.3))
        print("3 ok  {} fits 4 images: loss {:.4f} -> {:.4f}, angle error {:.1f} deg".format(arch, first, loss.item(), ang))

    # 4. attach to the symmetry model: its plane comes from the predictor, nothing else changes
    from scene.gaussian_predictor import GaussianSplatPredictor
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="default_config",
                      overrides=["+dataset=cars", "+experiment=lpips_100k", "model.symmetry.enabled=true"])
    model = GaussianSplatPredictor(cfg)
    net = model.network_with_offset
    pred = PlanePredictor(arch="small", pretrained=False)
    net.attach_plane_predictor(pred)
    sd_keys = set(model.state_dict().keys())
    assert not any(k.startswith("network_with_offset._ext") or "ext_plane" in k for k in sd_keys)
    N = cfg.data.training_resolution ** 2
    img = torch.rand(1, 1, 3, 128, 128)
    v2w = torch.eye(4).view(1, 1, 4, 4)
    quat = torch.tensor([[[1.0, 0, 0, 0]]])
    out = model(img, v2w, quat, None)
    assert torch.allclose(out["sym_plane"], pred(img[:, 0]), atol=1e-5)
    assert out["xyz"].shape[1] == 2 * N
    print("4 ok  attached predictor gives the model's plane; not in the state dict; 2N Gaussians")

    # 5. config route: model.symmetry.plane_predictor=<ckpt> builds the model without plane head/net
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "p.pth")
        save_predictor(pred, path)
        with initialize(version_base=None, config_path="../configs"):
            cfg2 = compose(config_name="default_config",
                           overrides=["+dataset=cars", "+experiment=lpips_100k", "model.symmetry.enabled=true",
                                      "model.symmetry.plane_predictor=" + path])
        m2 = GaussianSplatPredictor(cfg2)
        assert not hasattr(m2.network_with_offset, "plane_head") and m2.network_with_offset._ext_plane is not None
        out2 = m2(img, v2w, quat, None)
        assert torch.allclose(out2["sym_plane"], pred(img[:, 0]), atol=1e-5)
    print("5 ok  config route works; no unused parameters left for DDP")
    print("ALL PLANE PREDICTOR TESTS PASSED")


if __name__ == "__main__":
    main()
