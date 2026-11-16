"""
Level 1 (and evaluation for Levels 2/3): quality vs Gaussian budget for many selection criteria,
in ONE pass over the test set (one U-Net forward per object, all criteria x budgets reuse it).

Views of each test object (index 0 = conditioning view, 1..250 = novel views):
    odd  novel indices -> EVALUATION views (metrics)        -> subsample --n_eval_views
    even novel indices -> SCORING views (oracle importance)  -> subsample --n_score_views
so oracle scores never see the views they are evaluated on (no test leakage).

Output:
    <out>_objects.csv : one row per (object, criterion, budget): mean PSNR / masked PSNR / SSIM / LPIPS
    <out>_views.csv   : one row per (object, criterion, budget, view): + angle to the input camera

Example (20-object smoke test, ~5 min on a MIG slice):
    python eval_budget.py cars --max_objects 20 --out results/budget_smoke
Full run (all 704 objects; split across jobs with --start_idx / --max_objects if needed):
    python eval_budget.py cars --out results/budget_full
Learned head / fine-tuned model:
    python eval_budget.py cars --criteria opacity learned --head_ckpt runs/head_A/head.pth
    python eval_budget.py cars --experiment_path runs/budget_ft --criteria opacity learned ...
"""
import argparse
import csv
import math
import os
import time

import lpips as lpips_lib
import torch
import tqdm
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.dataset_factory import get_dataset
from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from utils.loss_utils import ssim as ssim_fn

import importance as imp

ORACLES = {"contribution", "error_sens", "error_sens_signed"}


def load_model(dataset_name, experiment_path, device):
    if experiment_path is None:
        cfg_path = hf_hub_download("szymanowiczs/splatter-image-v1", f"config_{dataset_name}.yaml")
        model_path = hf_hub_download("szymanowiczs/splatter-image-v1", f"model_{dataset_name}.pth")
    else:
        cfg_path = os.path.join(experiment_path, ".hydra", "config.yaml")
        model_path = os.path.join(experiment_path, "model_latest.pth")
    cfg = OmegaConf.load(cfg_path)
    model = GaussianSplatPredictor(cfg)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), cfg


def free_cached_example(dataset, example_id):
    """SRNDataset caches every image of every object it has loaded (self.all_rgbs etc.) and
    never frees them: ~49 MB per test object -> ~34 GB for 704 cars = the OOM kills we saw.
    Call this after each object (needs num_workers=0 so the cache lives in this process)."""
    for name, val in vars(dataset).items():
        if name.startswith("all_") and isinstance(val, dict):
            val.pop(example_id, None)


def worker_kwargs(num_workers):
    """DataLoader kwargs valid on old (1.x) and new PyTorch: prefetch only with workers."""
    if num_workers > 0:
        return dict(persistent_workers=True, prefetch_factor=4)
    return {}


class FreeCacheDataset(torch.utils.data.Dataset):
    """Wraps SRNDataset so that every item frees its image cache right after loading.
    Runs inside each DataLoader worker -> parallel file reading without the RAM leak."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        free_cached_example(self.ds, self.ds.get_example_id(i))
        return item


def model_input(data, cfg):
    n_in = cfg.data.input_images
    if cfg.data.origin_distances:
        return torch.cat([data["gt_images"][:, :n_in], data["origin_distances"][:, :n_in]], dim=2)
    return data["gt_images"][:, :n_in]


def even_subsample(idxs, n):
    if n <= 0 or n >= len(idxs):
        return list(idxs)
    step = len(idxs) / n
    return [idxs[int(i * step)] for i in range(n)]


def angle_deg(c0, c1):
    c0, c1 = c0 / c0.norm(), c1 / c1.norm()
    return math.degrees(math.acos(float((c0 * c1).sum().clamp(-1, 1))))


@torch.no_grad()
def main(a):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model, cfg = load_model(a.dataset_name, a.experiment_path, device)
    bg = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0],
                      dtype=torch.float32, device=device)
    lpips_net = lpips_lib.LPIPS(net="vgg").to(device)

    head, head_inputs = None, "all"
    if "learned" in a.criteria:
        head, head_inputs = imp.load_head(a.head_ckpt, device)   # input set stored in the checkpoint
        print(f"Loaded head {a.head_ckpt} (inputs={head_inputs})")

    dataset = get_dataset(cfg, a.split)
    end = len(dataset) if a.max_objects <= 0 else min(len(dataset), a.start_idx + a.max_objects)
    dataset_sub = torch.utils.data.Subset(FreeCacheDataset(dataset), list(range(a.start_idx, end)))
    loader = DataLoader(dataset_sub, batch_size=1, shuffle=False, num_workers=a.num_workers,
                        pin_memory=True, **worker_kwargs(a.num_workers))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f_obj = open(a.out + "_objects.csv", "w", newline="")
    f_view = open(a.out + "_views.csv", "w", newline="")
    w_obj, w_view = csv.writer(f_obj), csv.writer(f_view)
    w_obj.writerow(["object_idx", "example_id", "criterion", "budget", "n_gaussians",
                    "psnr", "psnr_masked", "ssim", "lpips", "select_ms"])
    w_view.writerow(["object_idx", "criterion", "budget", "view_id", "angle_deg",
                     "psnr", "psnr_masked", "ssim", "lpips"])

    for local_idx, data in enumerate(tqdm.tqdm(loader)):
        d_idx = a.start_idx + local_idx
        example_id = dataset.get_example_id(d_idx)
        data = {k: v.to(device) for k, v in data.items()}
        n_in = cfg.data.input_images
        inp = model_input(data, cfg)
        recon = model(inp, data["view_to_world_transforms"][:, :n_in],
                      data["source_cv2wT_quat"][:, :n_in], None)
        pc = {k: v[0].contiguous() for k, v in recon.items()}
        N = pc["xyz"].shape[0]

        novel = list(range(n_in, data["gt_images"].shape[1]))
        eval_ids = even_subsample(novel[0::2], a.n_eval_views)
        score_ids = even_subsample(novel[1::2], a.n_score_views)
        eval_cams = imp.cams_from_data(data, eval_ids)
        gts = torch.stack([data["gt_images"][0, i] for i in eval_ids])
        angles = [angle_deg(data["camera_centers"][0, 0], data["camera_centers"][0, i]) for i in eval_ids]

        def evaluate(pc_sel, crit, budget, n_kept, sel_ms):
            imgs = torch.stack([imp._render(pc_sel, c, bg, cfg) for c in eval_cams]).clamp(0, 1)
            mse = ((imgs - gts) ** 2).mean(dim=[1, 2, 3])
            psnr = -10 * torch.log10(mse)
            fg = (gts < 0.99).any(1, keepdim=True).float()            # object pixels of the GT
            mse_m = (((imgs - gts) ** 2) * fg).sum([1, 2, 3]) / (3 * fg.sum([1, 2, 3]).clamp_min(1))
            psnr_m = -10 * torch.log10(mse_m.clamp_min(1e-10))
            lp = lpips_net(imgs * 2 - 1, gts * 2 - 1).flatten()
            ss = torch.tensor([ssim_fn(imgs[v], gts[v]).item() for v in range(len(eval_ids))])
            for v, vid in enumerate(eval_ids):
                w_view.writerow([d_idx, crit, budget, vid, f"{angles[v]:.2f}", f"{psnr[v]:.4f}",
                                 f"{psnr_m[v]:.4f}", f"{ss[v]:.5f}", f"{lp[v]:.5f}"])
            w_obj.writerow([d_idx, example_id, crit, budget, n_kept, f"{psnr.mean():.4f}",
                            f"{psnr_m.mean():.4f}", f"{ss.mean():.5f}", f"{lp.mean():.5f}",
                            f"{sel_ms:.3f}"])

        evaluate(pc, "full", 1.0, N, 0.0)

        for crit in a.criteria:
            t0 = time.perf_counter()
            if crit == "merge":
                for b in (2, 4):
                    m = imp.merge_blocks(pc, b=b)
                    torch.cuda.synchronize()
                    evaluate(m, "merge", 1.0 / b ** 2, m["xyz"].shape[0],
                             1000 * (time.perf_counter() - t0))
                continue
            if crit == "contribution":
                score = imp.contribution_scores(pc, imp.cams_from_data(data, score_ids), cfg)
            elif crit in ("error_sens", "error_sens_signed"):
                sgts = [data["gt_images"][0, i] for i in score_ids]
                score = imp.error_sensitivity_scores(pc, imp.cams_from_data(data, score_ids), sgts,
                                                     bg, cfg, signed=crit.endswith("signed"))
            elif crit == "learned":
                x = imp.ImportanceHead.make_input(inp[:, 0], recon, inputs=head_inputs)
                score = head(x)[0]
            else:
                score = imp.cheap_scores(pc, crit, input_image=inp[0, 0], seed=d_idx)
            torch.cuda.synchronize()
            sel_ms = 1000 * (time.perf_counter() - t0)
            for budget in a.budgets:
                k = max(1, int(round(budget * N)))
                evaluate(imp.subset(pc, imp.topk_indices(score, k)), crit, budget, k, sel_ms)

        free_cached_example(dataset, example_id)
        if local_idx % 10 == 0:
            f_obj.flush(); f_view.flush()
    f_obj.close(); f_view.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset_name", default="cars")
    p.add_argument("--experiment_path", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--max_objects", type=int, default=0, help="0 = all")
    p.add_argument("--n_eval_views", type=int, default=25)
    p.add_argument("--n_score_views", type=int, default=25)
    p.add_argument("--criteria", nargs="+",
                   default=["random", "opacity", "scale", "opacity_area", "edge_opacity",
                            "contribution", "error_sens", "merge"])
    p.add_argument("--budgets", nargs="+", type=float,
                   default=[0.25, 0.20, 0.10, 0.05, 0.02, 0.01])
    p.add_argument("--head_ckpt", default=None)
    p.add_argument("--num_workers", type=int, default=4,
                   help="parallel file readers (BeeGFS is slow per file); keep <= --cpus-per-task")
    p.add_argument("--out", default="results/budget")
    main(p.parse_args())
