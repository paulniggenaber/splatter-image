"""
Train the stand-alone mirror-plane predictor (scene/plane_predictor.py).

Data: the disk cache of SRN cars (SRN_CACHE_DIR, filled by datasets/cache_srn.py).
Every car file holds 50 views with their cameras. For each view the true mirror plane
in that camera's frame follows from its world-to-view transform (x = 0 in dataset
coordinates, see datasets/srn.py). Each training step takes K random views of B cars.

    python train_plane.py --out experiments_out/plane1 --steps 4000

Reports, on validation cars (random views) and on test cars at the benchmark input
view (view 64): mean / median angle error, and the share of cars under 5 and 3 degrees.
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scene.plane_predictor import (PlanePredictor, matrix_loss, normal_from_raw,
                                   plane_from_normal, save_predictor)


def planes_from_wvt(wvt):
    """world_view_transforms (V, 4, 4) -> unit normals (V, 3) and offsets (V,)
    (same formula as datasets/srn.py)."""
    n = wvt[:, 0, :3].clone()
    d = (n * wvt[:, 3, :3]).sum(-1)
    return n, d


class CachedCars(Dataset):
    """One item = K random (or fixed) views of one cached car."""

    def __init__(self, folder, views_per_car, fixed_seed=None):
        self.files = sorted(glob.glob(os.path.join(folder, "*.pt")))
        assert len(self.files) > 0, "no cached cars in " + folder
        self.k = views_per_car
        self.fixed_seed = fixed_seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i])
        V = d["rgbs"].shape[0]
        if self.fixed_seed is None:
            idx = torch.randperm(V)[:self.k]
        else:
            g = torch.Generator().manual_seed(self.fixed_seed + i)
            idx = torch.randperm(V, generator=g)[:self.k]
        imgs = d["rgbs"][idx].float() / 255.0                       # (K, 3, H, W)
        n, dd = planes_from_wvt(d["world_view_transforms"][idx])
        return imgs, n, dd


def angle_deg(n_pred, n_true):
    cos = (n_pred * n_true).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(torch.acos(cos))


@torch.no_grad()
def evaluate_cached(model, loader, device):
    model.eval()
    errs, off = [], []
    for imgs, n, dd in loader:
        imgs = imgs.flatten(0, 1).to(device)
        n = n.flatten(0, 1).to(device)
        dd = dd.flatten(0, 1).to(device)
        plane = model(imgs)
        errs.append(angle_deg(plane[:, :3], n).cpu())
        sign = torch.where((plane[:, :3] * n).sum(-1) >= 0, 1.0, -1.0)
        off.append((plane[:, 3] * sign - dd).abs().cpu())
    return summarize(torch.cat(errs), torch.cat(off))


def summarize(err, off):
    return {"angle_mean": err.mean().item(), "angle_median": err.median().item(),
            "angle_p90": err.quantile(0.9).item(),
            "under5deg": (err < 5).float().mean().item(),
            "under3deg": (err < 3).float().mean().item(),
            "offset_mean": off.mean().item(), "n": int(err.numel())}


@torch.no_grad()
def evaluate_test(model, device, n_cars, workers):
    """Test cars at the benchmark input view (view 64), through the normal dataset class."""
    from hydra import compose, initialize
    from datasets.dataset_factory import get_dataset
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="default_config", overrides=["+dataset=cars", "+experiment=lpips_100k"])
    cfg.data.subset = n_cars
    dataset = get_dataset(cfg, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
    errs, off = [], []
    for data in loader:
        img = data["gt_images"][:, 0].to(device)                    # first image = the input view
        gt = data["sym_plane"].to(device)
        plane = model(img)
        errs.append(angle_deg(plane[:, :3], gt[:, :3]).cpu())
        sign = torch.where((plane[:, :3] * gt[:, :3]).sum(-1) >= 0, 1.0, -1.0)
        off.append((plane[:, 3] * sign - gt[:, 3]).abs().cpu())
    return summarize(torch.cat(errs), torch.cat(off))


@torch.no_grad()
def constant_baseline(train_ds, val_loader, device):
    """Best single plane for all images (principal axis of the mean of n n^T over
    training views). This is what 'no learning' achieves on random views."""
    acc = torch.zeros(3, 3)
    cnt = 0
    for i in range(0, len(train_ds), 4):
        _, n, _ = train_ds[i]
        acc += torch.einsum("bi,bj->ij", n, n)
        cnt += n.shape[0]
    _, vecs = torch.linalg.eigh(acc / cnt)
    n_const = vecs[:, -1].to(device)
    errs = []
    for imgs, n, dd in val_loader:
        n = n.flatten(0, 1).to(device)
        errs.append(angle_deg(n_const.unsqueeze(0).expand_as(n), n).cpu())
    e = torch.cat(errs)
    return {"angle_mean": e.mean().item(), "angle_median": e.median().item(),
            "under5deg": (e < 5).float().mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("SRN_CACHE_DIR"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "small"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--cars_per_step", type=int, default=8)
    ap.add_argument("--views_per_car", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--test_cars", type=int, default=100, help="0 = skip the test-car evaluation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")

    tag = glob.glob(os.path.join(args.cache, "srn_cars_*"))
    assert len(tag) == 1, "expected one cache folder, found {}".format(tag)
    train_ds = CachedCars(os.path.join(tag[0], "train"), args.views_per_car)
    val_ds = CachedCars(os.path.join(tag[0], "val"), 4, fixed_seed=123)
    train_loader = DataLoader(train_ds, batch_size=args.cars_per_step, shuffle=True, drop_last=True,
                              num_workers=args.workers, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=args.workers)
    print("train cars {}, val cars {}".format(len(train_ds), len(val_ds)), flush=True)

    model = PlanePredictor(arch=args.arch).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.05, anneal_strategy="cos")

    const = constant_baseline(train_ds, val_loader, device)
    print("constant-plane baseline on val (random views):", json.dumps(const), flush=True)

    step, t0, best = 0, time.time(), 1e9
    log = []
    while step < args.steps:
        for imgs, n, dd in train_loader:
            model.train()
            imgs = imgs.flatten(0, 1).to(device, non_blocking=True)
            n = n.flatten(0, 1).to(device, non_blocking=True)
            loss = matrix_loss(model.raw(imgs), n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print("[step {}] loss {:.5f} | {:.2f} s/step".format(
                    step, loss.item(), (time.time() - t0) / 50), flush=True)
                t0 = time.time()
            if step % args.eval_every == 0 or step == args.steps:
                s = evaluate_cached(model, val_loader, device)
                print("[val @ {}] angle mean {:.2f} median {:.2f} p90 {:.2f} | <5deg {:.1%} <3deg {:.1%}".format(
                    step, s["angle_mean"], s["angle_median"], s["angle_p90"], s["under5deg"], s["under3deg"]), flush=True)
                log.append({"step": step, **s})
                if s["angle_mean"] < best:
                    best = s["angle_mean"]
                    save_predictor(model, os.path.join(args.out, "plane_predictor.pth"),
                                   {"step": step, "val": s})
                t0 = time.time()
            if step >= args.steps:
                break

    from scene.plane_predictor import load_predictor
    best_model = load_predictor(os.path.join(args.out, "plane_predictor.pth"), device)
    result = {"constant_baseline_val": const, "val_log": log,
              "val_best": evaluate_cached(best_model, val_loader, device)}
    if args.test_cars > 0:
        result["test_view64"] = evaluate_test(best_model, device, args.test_cars, args.workers)
        print("[test, input view 64, {} cars]".format(args.test_cars), json.dumps(result["test_view64"]), flush=True)
    with open(os.path.join(args.out, "plane_scores.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("done. best val angle error {:.2f} deg".format(best), flush=True)


if __name__ == "__main__":
    main()
