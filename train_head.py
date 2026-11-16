"""
Level 2: train a tiny importance head on top of the FROZEN Splatter Image to imitate an
expensive multi-view oracle (knowledge distillation of a ranking).

Per step, for a training object:
    frozen model -> Gaussians -> oracle score from V random training views (computed on the fly)
    head(input image + predicted Gaussian maps) -> logits -> ranking loss vs oracle
At test time: one forward of the head, no renders, no ground truth.

    python train_head.py cars --target contribution --out runs/head_A     # variant A
    python train_head.py cars --target error_sens   --out runs/head_B     # variant B
Ablations (tune on val only; compare val_overlap10 in <out>/log.csv):
    --inputs rgb | gauss | all     what the head sees (default all)
    --loss   rank | mse            ranking vs regression on log-importance (default rank)
    --n_views 4 | 16 | 32          oracle views per training object
Speed: the bottleneck is reading many small files from BeeGFS, so data is loaded by
--num_workers parallel workers (set #SBATCH --cpus-per-task >= num_workers + 1), and the
validation objects + their oracle targets are computed ONCE at start-up and kept on the GPU.
Then:
    python eval_budget.py cars --criteria opacity opacity_area contribution learned --head_ckpt runs/head_A/head.pth
"""
import argparse
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Subset

import importance as imp
from datasets.dataset_factory import get_dataset
from eval_budget import FreeCacheDataset, load_model, model_input, worker_kwargs


def oracle(pc, data, view_ids, cfg, bg, target):
    cams = imp.cams_from_data(data, view_ids)
    if target == "contribution":
        s = imp.contribution_scores(pc, cams, cfg)
    else:
        gts = [data["gt_images"][0, i] for i in view_ids]
        s = imp.error_sensitivity_scores(pc, cams, gts, bg, cfg)
    return torch.log1p(s / (s.mean() + 1e-12))          # scale-free, heavy tail compressed


def topk_overlap(pred, target, frac):
    k = int(frac * pred.numel())
    a = set(torch.topk(pred, k).indices.tolist())
    b = set(torch.topk(target, k).indices.tolist())
    return len(a & b) / k


@torch.no_grad()
def build_val_cache(model, cfg, bg, a, device):
    """Load n_val validation objects once; keep head inputs (all 3 variants) + oracle targets on GPU."""
    val_set = get_dataset(cfg, "val")
    loader = DataLoader(Subset(FreeCacheDataset(val_set), list(range(a.n_val))), batch_size=1,
                        num_workers=a.num_workers)
    n_in, cache = cfg.data.input_images, []
    for vd in loader:
        vd = {k: v.to(device) for k, v in vd.items()}
        vin = model_input(vd, cfg)
        vrec = model(vin, vd["view_to_world_transforms"][:, :n_in], vd["source_cv2wT_quat"][:, :n_in], None)
        vpc = {k: v[0].contiguous() for k, v in vrec.items()}
        vids = list(range(n_in, vd["gt_images"].shape[1]))[1::10]          # 25 spread-out views
        cache.append(dict(x=imp.ImportanceHead.make_input(vin[:, 0], vrec, inputs=a.inputs),
                          target=oracle(vpc, vd, vids, cfg, bg, a.target),
                          opacity=vpc["opacity"].squeeze(-1)))
    return cache


def main(a):
    torch.manual_seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda:0")
    model, cfg = load_model(a.dataset_name, None, device)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg.opt.imgs_per_obj = 1 + a.n_views                 # input view + V oracle views per object
    bg = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0],
                      dtype=torch.float32, device=device)

    t0 = time.time()
    val_cache = build_val_cache(model, cfg, bg, a, device)
    print(f"validation cache: {len(val_cache)} objects in {time.time() - t0:.0f}s", flush=True)
    op10 = sum(topk_overlap(v["opacity"], v["target"], 0.10) for v in val_cache) / len(val_cache)

    train_set = get_dataset(cfg, "train")
    loader = DataLoader(FreeCacheDataset(train_set), batch_size=1, shuffle=True,
                        num_workers=a.num_workers, pin_memory=True,
                        **worker_kwargs(a.num_workers))
    head = imp.ImportanceHead(in_ch=imp.INPUT_CHANNELS[a.inputs]).to(device)
    meta = dict(inputs=a.inputs, loss=a.loss, target=a.target, n_views=a.n_views, lr=a.lr)
    print("head config:", meta, flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.steps // a.accum))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.csv"), "w")
    log.write("step,loss,val_overlap10,val_overlap5,opacity_overlap10\n")

    step, n_in = 0, cfg.data.input_images
    t_start = t_last = time.time()
    while step < a.steps:
        for data in loader:
            data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
            inp = model_input(data, cfg)
            with torch.no_grad():
                recon = model(inp, data["view_to_world_transforms"][:, :n_in],
                              data["source_cv2wT_quat"][:, :n_in], None)
            pc = {k: v[0].contiguous() for k, v in recon.items()}
            # frame 0 is the input, frames 1..V are random other views of the same object
            view_ids = list(range(n_in, data["gt_images"].shape[1]))
            target = oracle(pc, data, view_ids, cfg, bg, a.target)[None]
            logits = head(imp.ImportanceHead.make_input(inp[:, 0], recon, inputs=a.inputs))
            loss = imp.head_loss(logits, target, a.loss)
            (loss / a.accum).backward()
            if (step + 1) % a.accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1

            if step % a.print_every == 0:
                now = time.time()
                sps = a.print_every / (now - t_last); t_last = now
                eta = (a.steps - step) / max(sps, 1e-9) / 60
                print(f"step {step}/{a.steps} loss {loss.item():.4f} | {sps:.2f} steps/s | "
                      f"ETA {eta:.0f} min", flush=True)

            if step % a.val_every == 0 or step == a.steps:
                head.eval()
                with torch.no_grad():
                    ov10, ov5 = [], []
                    for v in val_cache:
                        vl = head(v["x"])[0]
                        ov10.append(topk_overlap(vl, v["target"], 0.10))
                        ov5.append(topk_overlap(vl, v["target"], 0.05))
                m = lambda x: sum(x) / len(x)
                print(f"[val] step {step} | top10% overlap head {m(ov10):.3f} (opacity {op10:.3f}) "
                      f"| top5% head {m(ov5):.3f} | elapsed {(time.time() - t_start) / 60:.1f} min",
                      flush=True)
                log.write(f"{step},{loss.item():.5f},{m(ov10):.4f},{m(ov5):.4f},{op10:.4f}\n")
                log.flush()
                imp.save_head(head, os.path.join(a.out, "head.pth"), **meta)
                head.train()
            if step >= a.steps:
                break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset_name", default="cars")
    p.add_argument("--target", choices=["contribution", "error_sens"], default="contribution")
    p.add_argument("--inputs", choices=["all", "rgb", "gauss"], default="all")
    p.add_argument("--loss", choices=["rank", "mse"], default="rank")
    p.add_argument("--n_views", type=int, default=16, help="oracle views per training object")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=6,
                   help="parallel file readers; set #SBATCH --cpus-per-task >= num_workers + 1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/head_A")
    main(p.parse_args())
