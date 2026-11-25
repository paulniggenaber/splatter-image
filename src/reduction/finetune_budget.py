# entirely new code file from Taya
"""
Level 3 (budget-aware fine-tuning) and Level 3b (viewpoint-robust fine-tuning) of Splatter Image.

Starts from the released cars checkpoint and fine-tunes the reconstructor with:
  --mask none      plain fine-tuning (CONTROL run)
  --mask learned   straight-through top-k mask from the importance head (Level 3, head trained jointly)
  --mask hybrid    hard top-k mask from the free hybrid rank (Level 3 ablation: is a learned mask needed?)
  --keep 0.10      kept fraction of Gaussians for the mask
  --view_weighting none | fixed | dro      per-view loss weights by input->target angle (Level 3b)

Why not train_network.py: it validates on all 352 val objects (~1 h), logs to wandb and keeps an EMA
with beta 0.9999 (updated every 10 steps) - over a short fine-tune the EMA barely moves away from the
start weights, so the fine-tuning would be invisible. Here: no EMA, fast cached validation on 20 val
objects, resume after a time limit, outputs readable by eval_budget.py --experiment_path.

Examples
  # smoke test (~5-10 min)
  python -m src.reduction.finetune_budget --mask learned --keep 0.10 --head_ckpt runs/abl_reference/head.pth \
      --max_iters 200 --val_every 100 --out runs/ft_smoke
  # main runs (10 h each)
  python -m src.reduction.finetune_budget --mask none                      --hours 10 --out runs/ft_ctrl
  python -m src.reduction.finetune_budget --mask learned --keep 0.10 --head_ckpt runs/abl_reference/head.pth --hours 10 --out runs/ft10_learned
  python -m src.reduction.finetune_budget --mask hybrid  --keep 0.10       --hours 10 --out runs/ft10_hybrid
  python -m src.reduction.finetune_budget --mask none --view_weighting dro --hours 10 --out runs/ft_dro
Evaluation (704 test cars)
  python -m src.reduction.eval_budget cars --experiment_path runs/ft10_learned --criteria opacity hybrid learned \
      --head_ckpt runs/ft10_learned/head.pth --out results/test_ft10_learned
"""
import argparse
import copy
import math
import os
import random
import time

import lpips as lpips_lib
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.reduction import importance as imp
from datasets.dataset_factory import get_dataset
from src.reduction.eval_budget import FreeCacheDataset, load_model, model_input, worker_kwargs

BIN_DEG = 30
N_BINS = 180 // BIN_DEG


# ----------------------------------------------------------------------------- helpers
def angle_bins(centers):
    """centers (B, F, 3) -> angle (deg) between camera 0 (input) and every camera, and its 30-degree bin."""
    c = centers / centers.norm(dim=-1, keepdim=True)
    cos = (c[:, :1] * c).sum(-1).clamp(-1, 1)
    ang = torch.rad2deg(torch.acos(cos))
    return ang, (ang // BIN_DEG).long().clamp(max=N_BINS - 1)


def psnr(img, gt):
    return -10 * torch.log10(((img - gt) ** 2).mean(dim=[-1, -2, -3]).clamp_min(1e-10))


class EMA:
    """Exponential moving average of the weights (model and head). Validation and saving use the EMA
    copy: the raw weights oscillate by ~1 dB between validations (seen in the lr check)."""
    def __init__(self, module, beta):
        self.beta = beta
        self.shadow = copy.deepcopy(module).eval()
        for q in self.shadow.parameters():
            q.requires_grad_(False)

    @torch.no_grad()
    def update(self, module):
        # one fused op over all tensors (a python loop over hundreds of tensors costs ~30 % of a step)
        ep = [e for e in self.shadow.parameters()]
        mp = [q.detach() for q in module.parameters()]
        torch._foreach_mul_(ep, self.beta)
        torch._foreach_add_(ep, mp, alpha=1 - self.beta)
        for e, q in zip(self.shadow.buffers(), module.buffers()):
            e.copy_(q)


def apply_mask(recon, rgb, a, head, head_inputs, keep=None, mask=None):
    """Returns recon with masked opacities (batched dict B x N x ...). keep=None -> a.keep; keep>=1 -> no mask."""
    keep = a.keep if keep is None else keep
    # symmetric model: drop non-Gaussian keys; the budget counts ORIGINAL (stored) Gaussians,
    # each kept original keeps its mirrored twin (twins are not stored: plane + switch rebuild them)
    recon, n_orig, sym = imp.split_symmetric(recon)
    mask = a.mask if mask is None else mask
    if mask == "none" or keep >= 1.0:
        return recon
    orig = imp.originals(recon, n_orig)
    if mask == "learned":
        logits = head(imp.ImportanceHead.make_input(rgb, orig, inputs=head_inputs))
        m = imp.straight_through_topk_mask(logits, keep, tau=a.tau)
    else:  # hybrid / random: hard mask, no gradient through the selection; the reconstructor adapts to it
        B, N = orig["opacity"].shape[:2]
        k = max(1, int(round(keep * N)))
        m = torch.zeros(B, N, device=rgb.device)
        for b in range(B):
            if mask == "random":   # ablation: content-independent selection, a new random subset every step
                idx = torch.randperm(N, device=rgb.device)[:k]
            else:
                pc = {kk: v[b].detach() for kk, v in orig.items()}
                idx = imp.topk_indices(imp.cheap_scores(pc, "hybrid"), k)
            m[b, idx] = 1.0
    if sym:
        m = torch.cat([m, m], dim=1)          # twin of i sits at n_orig + i
    out = dict(recon)
    out["opacity"] = recon["opacity"] * m.unsqueeze(-1)
    return out


class ViewWeighter:
    """Per-image loss weights from the input->target angle bin.
    fixed: weight `fixed_w` for the 90-120 degree bin; dro: softmax over running bin losses
    (Group-DRO-style, Sagawa et al. 2020): q_g ~ exp(eta * L_g / mean(L)), L_g = EMA of the bin's L2 loss."""
    def __init__(self, a, device):
        self.a = a
        self.ema = torch.full((N_BINS,), float("nan"), device=device)

    def state(self):
        return {"ema": self.ema}

    def load(self, s):
        self.ema = s["ema"].to(self.ema.device)

    def bin_weights(self):
        if self.a.view_weighting == "fixed":
            w = torch.ones(N_BINS, device=self.ema.device)
            w[90 // BIN_DEG] = self.a.fixed_w
            return w
        if self.a.view_weighting == "dro":
            L = torch.nan_to_num(self.ema, nan=float(torch.nanmean(self.ema)) if torch.isfinite(self.ema).any() else 1.0)
            q = torch.exp(self.a.dro_eta * L / L.mean().clamp_min(1e-12))
            return (q / q.mean()).clamp(min=self.a.dro_wmin, max=self.a.dro_wmax)
        return torch.ones(N_BINS, device=self.ema.device)

    def __call__(self, bins, l2_per_img, is_input=None):
        with torch.no_grad():
            stat_bins = bins if is_input is None else bins[~is_input]
            stat_l2 = l2_per_img if is_input is None else l2_per_img[~is_input]
            for g in stat_bins.unique():
                v = stat_l2[stat_bins == g].mean()
                cur = self.ema[g]
                self.ema[g] = v if torch.isnan(cur) else self.a.dro_beta * cur + (1 - self.a.dro_beta) * v
        w = self.bin_weights()[bins]
        if is_input is not None and self.a.view_weighting != "none":
            w = torch.where(is_input, torch.ones_like(w), w)   # input view keeps weight 1 (it is not a "viewpoint")
        return w / w.mean()          # mean weight 1 per batch -> learning rate stays comparable


# ----------------------------------------------------------------------------- validation
@torch.no_grad()
def build_val_cache(cfg, a, device):
    val = get_dataset(cfg, "val")
    loader = DataLoader(Subset(FreeCacheDataset(val), list(range(a.n_val))), batch_size=1,
                        num_workers=a.num_workers)
    cache = []
    for vd in loader:
        F = vd["gt_images"].shape[1]
        ids = [0] + list(range(1, F))[1::10]                 # input view + 25 spread-out novel views
        # slice only per-view tensors (B, F, ...); keep per-object fields as they are
        # (e.g. the symmetry branch's dataset also returns a per-object plane, shape (B, 4))
        cache.append({k: (v[:, ids] if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == F else v).to(device)
                      for k, v in vd.items()})
    return cache


@torch.no_grad()
def validate(model, head, head_inputs, cache, cfg, bg, a):
    model.eval()
    if head is not None:
        head.eval()
    budgets = a.keep_choices if a.keep_choices else [a.keep]
    full_all, bins_all = [], []
    masked_by = {kk: [] for kk in budgets if kk < 1.0}
    for vd in cache:
        inp = model_input(vd, cfg)
        recon = model(inp, vd["view_to_world_transforms"][:, :1], vd["source_cv2wT_quat"][:, :1], None)
        variants = {"full": imp.split_symmetric(recon)[0]}
        vmask = a.mask if a.mask != "none" else a.val_mask
        if vmask != "none":
            for kk in budgets:
                if kk < 1.0:
                    variants[kk] = apply_mask(recon, inp[:, 0], a, head, head_inputs, keep=kk, mask=vmask)
        _, bins = angle_bins(vd["camera_centers"])
        for name, rec in variants.items():
            pc = {k: v[0].contiguous() for k, v in rec.items()}
            imgs = torch.stack([imp._render(pc, (vd["world_view_transforms"][0, r], vd["full_proj_transforms"][0, r],
                                                 vd["camera_centers"][0, r]), bg, cfg)
                                for r in range(1, vd["gt_images"].shape[1])]).clamp(0, 1)
            p = psnr(imgs, vd["gt_images"][0, 1:])
            (full_all if name == "full" else masked_by[name]).append(p)
        bins_all.append(bins[0, 1:])
    full = torch.cat(full_all); bins = torch.cat(bins_all)
    per_bin = {f"{g * BIN_DEG}-{(g + 1) * BIN_DEG}": full[bins == g].mean().item()
               for g in range(N_BINS) if (bins == g).any()}
    out = {"psnr_full": full.mean().item(), "psnr_worst_bin": min(per_bin.values()), "per_bin": per_bin}
    if masked_by:
        per_budget = {kk: torch.cat(v).mean().item() for kk, v in masked_by.items()}
        if 1.0 in budgets:
            per_budget[1.0] = out["psnr_full"]
        out["per_budget"] = per_budget
        # one fixed budget: PSNR at that budget; random budgets: mean over the trained budgets
        out["psnr_masked"] = sum(per_budget.values()) / len(per_budget)
    return out


# ----------------------------------------------------------------------------- main
def main(a):
    torch.manual_seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    os.makedirs(a.out, exist_ok=True)

    if a.decay == "cosine" and a.max_iters <= 0:
        print("WARNING: --decay cosine needs --max_iters; using a constant lr after warm-up", flush=True)
    model, cfg = load_model("cars", a.init_path, device, a.init_ckpt)   # released model, or e.g. Paul's sym_long
    if cfg.model.get("symmetry", None) is not None and cfg.model.symmetry.get("enabled", False):
        print("starting from a SYMMETRY model: budget = fraction of ORIGINAL Gaussians, twins follow", flush=True)
    cfg.opt.imgs_per_obj = a.imgs_per_obj
    cfg.opt.batch_size = a.batch_size
    os.makedirs(os.path.join(a.out, ".hydra"), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(a.out, ".hydra", "config.yaml"))   # for eval_budget --experiment_path
    bg = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0], dtype=torch.float32, device=device)

    head, head_inputs = None, "all"
    params = [{"params": model.parameters(), "lr": a.lr}]
    if a.mask == "learned":
        head, head_inputs = imp.load_head(a.head_ckpt, device)
        head.train()
        params.append({"params": head.parameters(), "lr": a.head_lr})
    opt = torch.optim.Adam(params, eps=1e-15, betas=(0.9, 0.999))
    # linear warm-up: a fresh Adam state with eps=1e-15 makes the first steps large
    def lr_factor(i):
        if i < a.warmup:
            return (i + 1) / max(1, a.warmup)
        if a.decay == "cosine" and a.max_iters > a.warmup:
            t = min(1.0, (i - a.warmup) / (a.max_iters - a.warmup))
            return a.min_lr_factor + (1 - a.min_lr_factor) * 0.5 * (1 + math.cos(math.pi * t))
        return 1.0
    # in the first runs val PSNR peaked after ~2k it and then drifted down with a constant lr
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    ema_model = EMA(model, a.ema)
    ema_head = EMA(head, a.ema) if head is not None else None
    lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval() if a.lambda_lpips > 0 else None
    for p in (lpips_fn.parameters() if lpips_fn is not None else []):
        p.requires_grad_(False)
    weighter = ViewWeighter(a, device)

    # ---- resume
    it, best = 0, -1.0
    ck = os.path.join(a.out, "train_state.pth")
    if os.path.isfile(ck):
        s = torch.load(ck, map_location=device)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        ema_model.shadow.load_state_dict(s["ema_model"])
        if ema_head is not None:
            ema_head.shadow.load_state_dict(s["ema_head"])
        if head is not None:
            head.load_state_dict(s["head"])
        weighter.load(s["weighter"]); it, best = s["iteration"], s["best"]
        print(f"resumed from iteration {it}", flush=True)
    meta = dict(inputs=head_inputs, loss="finetune", target="rendering", lr=a.head_lr)

    def save(tag):
        torch.save({"iteration": it, "best_PSNR": best, "model_state_dict": ema_model.shadow.state_dict()},
                   os.path.join(a.out, f"model_{tag}.pth"))
        if head is not None:
            imp.save_head(ema_head.shadow, os.path.join(a.out, f"head_{tag}.pth" if tag != "latest" else "head.pth"), **meta)

    def save_state():
        torch.save({"iteration": it, "best": best, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "ema_model": ema_model.shadow.state_dict(),
                    "ema_head": ema_head.shadow.state_dict() if ema_head is not None else None,
                    "head": head.state_dict() if head is not None else None,
                    "weighter": weighter.state()}, ck)

    t0 = time.time()
    val_cache = build_val_cache(cfg, a, device)
    print(f"validation cache: {len(val_cache)} objects in {time.time() - t0:.0f}s", flush=True)
    log = open(os.path.join(a.out, "log.csv"), "a")
    if it == 0:
        log.write("iteration,loss,psnr_full,psnr_masked,psnr_worst_bin,per_bin\n")
        v = validate(model, head, head_inputs, val_cache, cfg, bg, a)
        if "per_budget" in v and a.keep_choices:
            print("[val]   per budget: " + " ".join(f"{k * 100:g}%={x:.3f}" for k, x in sorted(v["per_budget"].items())), flush=True)
        print(f"[val] start | full {v['psnr_full']:.3f} | masked {v.get('psnr_masked', float('nan')):.3f} "
              f"| worst bin {v['psnr_worst_bin']:.3f} | {v['per_bin']}", flush=True)
        log.write(f"0,nan,{v['psnr_full']:.4f},{v.get('psnr_masked', float('nan')):.4f},"
                  f"{v['psnr_worst_bin']:.4f},\"{v['per_bin']}\"\n"); log.flush()

    train = get_dataset(cfg, "train")
    loader = DataLoader(FreeCacheDataset(train), batch_size=a.batch_size, shuffle=True, drop_last=True,
                        num_workers=a.num_workers, pin_memory=True, **worker_kwargs(a.num_workers))
    model.train()
    t_start = t_last = time.time()
    deadline = t_start + a.hours * 3600 if a.hours > 0 else math.inf
    done = False
    while not done:
        for data in loader:
            data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
            inp = model_input(data, cfg)
            recon = model(inp, data["view_to_world_transforms"][:, :1], data["source_cv2wT_quat"][:, :1], None)
            keep_now = random.choice(a.keep_choices) if a.keep_choices else a.keep
            recon = apply_mask(recon, inp[:, 0], a, head, head_inputs, keep=keep_now)
            ang, bins = angle_bins(data["camera_centers"])

            imgs, gts, bsel, asel = [], [], [], []
            for b in range(inp.shape[0]):
                pc = {k: v[b].contiguous() for k, v in recon.items()}
                for r in range(1, data["gt_images"].shape[1]):   # frame 0 = input; as in train_network.py
                    imgs.append(imp._render(pc, (data["world_view_transforms"][b, r], data["full_proj_transforms"][b, r],
                                                 data["camera_centers"][b, r]), bg, cfg))
                    gts.append(data["gt_images"][b, r]); bsel.append(bins[b, r]); asel.append(ang[b, r])
            imgs, gts, bsel = torch.stack(imgs), torch.stack(gts), torch.stack(bsel)
            is_input = torch.stack(asel) < 1.0          # the re-rendered input view (angle ~0)
            l2 = ((imgs - gts) ** 2).mean(dim=[1, 2, 3])
            per_img = (1 - a.lambda_lpips) * l2
            if lpips_fn is not None:
                per_img = per_img + a.lambda_lpips * lpips_fn(imgs * 2 - 1, gts * 2 - 1).flatten()
            w = weighter(bsel, l2.detach(), is_input)
            loss = (w * per_img).mean()
            if a.sparsity_l1 > 0:
                # soft "training for sparsity" baseline: push unneeded Gaussians towards transparency,
                # prune post-hoc afterwards (compare with the hard top-k budget of --mask)
                loss = loss + a.sparsity_l1 * recon["opacity"].mean()
            assert torch.isfinite(loss), "non-finite loss"
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema_model.update(model)
            if ema_head is not None:
                ema_head.update(head)
            it += 1

            if it % a.print_every == 0:
                now = time.time(); ips = a.print_every / (now - t_last); t_last = now
                left = min(deadline - now, (a.max_iters - it) / max(ips, 1e-9) if a.max_iters > 0 else math.inf)
                msg = f"it {it} loss {loss.item():.5f} | {ips:.2f} it/s | left {left / 60:.0f} min"
                if a.sparsity_l1 > 0:
                    msg += f" | mean opacity {recon['opacity'].mean().item():.4f}"
                if a.view_weighting != "none":
                    msg += " | bin w " + " ".join(f"{x:.2f}" for x in weighter.bin_weights().tolist())
                print(msg, flush=True)

            stop = (a.max_iters > 0 and it >= a.max_iters) or time.time() > deadline
            if it % a.val_every == 0 or stop:
                v = validate(ema_model.shadow, ema_head.shadow if ema_head is not None else None,
                             head_inputs, val_cache, cfg, bg, a)
                score = v["psnr_masked"] if (a.mask != "none" or a.val_mask != "none") else (
                    v["psnr_worst_bin"] if a.view_weighting != "none" else v["psnr_full"])
                if "per_budget" in v and a.keep_choices:
                    print("[val]   per budget: " + " ".join(f"{k * 100:g}%={x:.3f}" for k, x in sorted(v["per_budget"].items())), flush=True)
                print(f"[val] it {it} | full {v['psnr_full']:.3f} | masked {v.get('psnr_masked', float('nan')):.3f} "
                      f"| worst bin {v['psnr_worst_bin']:.3f} | elapsed {(time.time() - t_start) / 3600:.2f} h", flush=True)
                log.write(f"{it},{loss.item():.6f},{v['psnr_full']:.4f},{v.get('psnr_masked', float('nan')):.4f},"
                          f"{v['psnr_worst_bin']:.4f},\"{v['per_bin']}\"\n"); log.flush()
                if score > best:
                    best = score; save("best")
                save("latest"); save_state()
            if stop:
                done = True
                break
    print(f"finished at iteration {it}; best val score {best:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mask", choices=["none", "learned", "hybrid", "random"], default="none")
    p.add_argument("--sparsity_l1", type=float, default=0.0, help="soft sparsity baseline: + lambda * mean opacity")
    p.add_argument("--val_mask", choices=["none", "hybrid"], default="none",
                   help="with --mask none: validate after post-hoc pruning with this rule at --keep")
    p.add_argument("--init_path", default=None, help="run dir to start from (default: released model), e.g. Paul's sym_long")
    p.add_argument("--init_ckpt", default="model_latest.pth")
    p.add_argument("--keep", type=float, default=0.10)
    p.add_argument("--keep_choices", nargs="+", type=float, default=None,
                   help="random budget per step, e.g. 0.05 0.1 0.2 1.0 (nested hybrid masks -> one model for all budgets)")
    p.add_argument("--head_ckpt", default="runs/abl_reference/head.pth")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--view_weighting", choices=["none", "fixed", "dro"], default="none")
    p.add_argument("--fixed_w", type=float, default=2.0, help="weight of the 90-120 degree bin (fixed)")
    p.add_argument("--dro_eta", type=float, default=1.0, help="DRO temperature on relative bin losses")
    p.add_argument("--dro_beta", type=float, default=0.99, help="EMA factor of the bin losses")
    p.add_argument("--dro_wmax", type=float, default=5.0, help="cap on a bin weight (before batch normalisation)")
    p.add_argument("--dro_wmin", type=float, default=0.2, help="floor on a bin weight, so every viewpoint keeps being trained")
    p.add_argument("--lr", type=float, default=5e-6, help="reconstructor lr (stage-2 lr of the paper)")
    p.add_argument("--head_lr", type=float, default=1e-5, help="lower than for Level 2: a fast-moving head flips the mask every step")
    p.add_argument("--warmup", type=int, default=500, help="linear lr warm-up iterations")
    p.add_argument("--decay", choices=["none", "cosine"], default="cosine", help="cosine needs --max_iters")
    p.add_argument("--min_lr_factor", type=float, default=0.05, help="final lr = min_lr_factor * lr (cosine)")
    p.add_argument("--ema", type=float, default=0.999, help="EMA of weights used for validation/saving (horizon ~1000 it)")
    p.add_argument("--lambda_lpips", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--imgs_per_obj", type=int, default=4, help="as in the original config (input + targets)")
    p.add_argument("--hours", type=float, default=0, help="wall-clock budget; 0 = no limit")
    p.add_argument("--max_iters", type=int, default=0, help="0 = no limit")
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ft")
    main(p.parse_args())
