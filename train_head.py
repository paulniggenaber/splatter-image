"""
Level 2: train a tiny importance head on top of the FROZEN Splatter Image to imitate an
expensive multi-view oracle (knowledge distillation of a ranking).

Per step, for a training object:
    frozen model -> Gaussians -> oracle score from V random training views (computed on the fly)
    head(input image + predicted Gaussian maps) -> logits -> pairwise ranking loss vs oracle
At test time: one forward of the head, no renders, no ground truth.

    python train_head.py cars --target contribution --out runs/head_A     # variant A
    python train_head.py cars --target error_sens   --out runs/head_B     # variant B
Then:
    python eval_budget.py cars --criteria opacity contribution learned --head_ckpt runs/head_A/head.pth
"""
import argparse
import os
import random

import torch
from torch.utils.data import DataLoader

import importance as imp
from datasets.dataset_factory import get_dataset
from eval_budget import free_cached_example, load_model, model_input


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


def main(a):
    torch.manual_seed(a.seed); random.seed(a.seed)
    device = torch.device("cuda:0")
    model, cfg = load_model(a.dataset_name, None, device)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg.opt.imgs_per_obj = 1 + a.n_views                 # input view + V oracle views per object
    bg = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0],
                      dtype=torch.float32, device=device)

    train_set, val_set = get_dataset(cfg, "train"), get_dataset(cfg, "val")
    loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    head = imp.ImportanceHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.csv"), "w")
    log.write("step,loss,val_overlap10,val_overlap5,opacity_overlap10\n")

    step, n_in = 0, cfg.data.input_images
    while step < a.steps:
        for idx_in_epoch, data in enumerate(loader):
            data = {k: v.to(device) for k, v in data.items()}
            inp = model_input(data, cfg)
            with torch.no_grad():
                recon = model(inp, data["view_to_world_transforms"][:, :n_in],
                              data["source_cv2wT_quat"][:, :n_in], None)
            pc = {k: v[0].contiguous() for k, v in recon.items()}
            # frame 0 is the input, frames 1..V are random other views of the same object
            view_ids = list(range(n_in, data["gt_images"].shape[1]))
            target = oracle(pc, data, view_ids, cfg, bg, a.target)[None]
            logits = head(imp.ImportanceHead.make_input(inp[:, 0], recon))
            loss = imp.ranking_loss(logits, target)
            (loss / a.accum).backward()
            if (step + 1) % a.accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
            sched.step(); step += 1
            for ds in (train_set,):                          # keep RAM flat
                for name, val in vars(ds).items():
                    if name.startswith("all_") and isinstance(val, dict) and len(val) > 8:
                        val.clear()

            if step % a.val_every == 0 or step == a.steps:
                head.eval()
                ov10, ov5, op10 = [], [], []
                for j in range(a.n_val):
                    vd = {k: (v[None] if torch.is_tensor(v) else v) for k, v in val_set[j].items()}
                    vd = {k: v.to(device) for k, v in vd.items()}
                    vin = model_input(vd, cfg)
                    with torch.no_grad():
                        vrec = model(vin, vd["view_to_world_transforms"][:, :n_in],
                                     vd["source_cv2wT_quat"][:, :n_in], None)
                        vpc = {k: v[0].contiguous() for k, v in vrec.items()}
                        vids = list(range(n_in, vd["gt_images"].shape[1]))[1::10]  # 25 views
                        vt = oracle(vpc, vd, vids, cfg, bg, a.target)
                        vl = head(imp.ImportanceHead.make_input(vin[:, 0], vrec))[0]
                    ov10.append(topk_overlap(vl, vt, 0.10)); ov5.append(topk_overlap(vl, vt, 0.05))
                    op10.append(topk_overlap(vpc["opacity"].squeeze(-1), vt, 0.10))
                    free_cached_example(val_set, val_set.get_example_id(j))
                m = lambda x: sum(x) / len(x)
                print(f"step {step} loss {loss.item():.4f} | top10% overlap head {m(ov10):.3f} "
                      f"(opacity {m(op10):.3f}) | top5% head {m(ov5):.3f}", flush=True)
                log.write(f"{step},{loss.item():.5f},{m(ov10):.4f},{m(ov5):.4f},{m(op10):.4f}\n")
                log.flush()
                torch.save(head.state_dict(), os.path.join(a.out, "head.pth"))
                head.train()
            if step >= a.steps:
                break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset_name", default="cars")
    p.add_argument("--target", choices=["contribution", "error_sens"], default="contribution")
    p.add_argument("--n_views", type=int, default=16, help="oracle views per training object")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/head_A")
    main(p.parse_args())
