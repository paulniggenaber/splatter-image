# entirely new code file from Taya
"""
Cost benchmark + qualitative 360-degree comparison video (paper: Sec. Cost; poster / defense: video).

What it measures, for several representations of the same objects:
    full model (16 384 Gaussians) | opacity 25 % | opacity 10 % | hybrid 10 % | hybrid 5 | merge 4x4 (6.25 %)
    [+ fine-tuned model (Level 3) with its hybrid 10 % mask, if --ft_path is given]
  1. render time of ONE frame at 128^2 / 256^2 / 512^2 output resolution
     (CUDA events, warm-up, mean +- std over --reps repetitions, averaged over --n_timing objects)
  2. peak extra GPU memory while rendering (torch.cuda.max_memory_allocated minus the memory before)
  3. time to render a 100-frame 360-degree sequence (the 250 test-orbit cameras) at 128^2 and 512^2
  4. stage times for context: U-Net forward (reconstruction) and selection
  5. storage (23 floats per Gaussian)
and it renders a side-by-side 360-degree video and a 3-view image strip for one object.

    python -m src.reduction.benchmark_cost --ft_path runs/ft10_hybrid --out results/cost
Outputs (in --out):
    cost_table.csv / cost_table.md      everything above, one row per (variant, resolution)
    video_obj<k>.mp4 (or .gif)           side-by-side 360-degree video, labels + #Gaussians under each panel
    frames_obj<k>/                       the PNG frames (to rebuild the mp4 on a laptop with ffmpeg)
    strip_obj<k>.png                     3 viewpoints x panels, for the paper / poster
Notes: run on a full A100 (gres gpu:7g.80gb) for paper-quality timings; on a MIG slice timings are
indicative only (the script prints the GPU name into the table).
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

from src.reduction import importance as imp
from datasets.dataset_factory import get_dataset
from src.reduction.eval_budget import FreeCacheDataset, load_model, model_input

BYTES_PER_GAUSSIAN = 23 * 4


# ----------------------------------------------------------------------------- helpers
def cfg_at(cfg, res):
    c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    c.data.training_resolution = int(res)      # render_predicted uses this as output size (fov fixed)
    return c


def cam(data, r):
    return (data["world_view_transforms"][0, r], data["full_proj_transforms"][0, r], data["camera_centers"][0, r])


def select(pc, kind, keep):
    if kind == "full":
        return pc
    if kind == "merge":
        return imp.merge_blocks(pc, b=4)
    k = max(1, int(round(keep * pc["xyz"].shape[0])))
    return imp.subset(pc, imp.topk_indices(imp.cheap_scores(pc, kind), k))


def cuda_time(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.mean(times)), float(np.std(times))


def to_uint8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def label_panel(arr, title, sub):
    """Add a white caption band under an RGB uint8 panel."""
    h, w, _ = arr.shape
    band = 34 if w >= 256 else 26
    canvas = Image.new("RGB", (w, h + band), (255, 255, 255))
    canvas.paste(Image.fromarray(arr), (0, 0))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13 if w >= 256 else 10)
    except OSError:
        font = ImageFont.load_default()
    d.text((4, h + 2), title, fill=(20, 20, 20), font=font)
    d.text((4, h + band // 2), sub, fill=(90, 90, 90), font=font)
    return np.asarray(canvas)


def write_video(frames, path_stem, fps):
    try:
        import imageio
        imageio.mimsave(path_stem + ".mp4", frames, fps=fps, quality=8, macro_block_size=1)
        return path_stem + ".mp4"
    except Exception as ex:                                 # no ffmpeg backend -> GIF with Pillow
        print(f"mp4 failed ({ex.__class__.__name__}); writing GIF instead", flush=True)
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(path_stem + ".gif", save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
        return path_stem + ".gif"


# ----------------------------------------------------------------------------- main
@torch.no_grad()
def main(a):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    os.makedirs(a.out, exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    print("GPU:", gpu, flush=True)

    model, cfg = load_model("cars", None, device)
    ft_model = None
    if a.ft_path:
        ft_model, _ = load_model("cars", a.ft_path, device, a.ft_ckpt)
    bg = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0], dtype=torch.float32, device=device)

    variants = [("Full model", "base", "full", 1.0),
                ("Opacity 25%", "base", "opacity", 0.25),
                ("Opacity 10%", "base", "opacity", 0.10),
                ("Hybrid 10%", "base", "hybrid", 0.10),
                ("Hybrid 5%", "base", "hybrid", 0.05),
                ("Merge 4x4 (6.25%)", "base", "merge", 0.0625)]
    if ft_model is not None:
        variants.append((f"Fine-tuned + hybrid {int(round(a.ft_keep * 100))}% (ours)", "ft", "hybrid", a.ft_keep))

    test = get_dataset(cfg, "test")
    ids = sorted(set(list(range(a.n_timing)) + [a.video_obj]))
    loader = DataLoader(Subset(FreeCacheDataset(test), ids), batch_size=1, num_workers=a.num_workers)

    rows = {}           # (variant, res) -> lists
    seq = {}            # (variant, res) -> list of 100-frame times
    stage = {"unet_ms": [], "select_hybrid_ms": [], "select_merge_ms": []}
    video_data = None
    for obj_i, data in zip(ids, loader):
        data = {k: v.to(device) for k, v in data.items()}
        inp = model_input(data, cfg)
        args = (inp, data["view_to_world_transforms"][:, :1], data["source_cv2wT_quat"][:, :1], None)
        if obj_i < a.n_timing:
            stage["unet_ms"].append(cuda_time(lambda: model(*args), 10, 3)[0])
        recon = {"base": {k: v[0].contiguous() for k, v in model(*args).items()}}
        if ft_model is not None:
            recon["ft"] = {k: v[0].contiguous() for k, v in ft_model(*args).items()}
        if obj_i < a.n_timing:
            stage["select_hybrid_ms"].append(cuda_time(lambda: select(recon["base"], "hybrid", 0.1), 10, 3)[0])
            stage["select_merge_ms"].append(cuda_time(lambda: select(recon["base"], "merge", 0.0625), 5, 2)[0])

        sets = {name: select(recon[src], kind, keep) for name, src, kind, keep in variants}
        if obj_i < a.n_timing:
            r_view = 1 + (data["gt_images"].shape[1] - 1) // 2            # one novel view (side-ish)
            for res in a.resolutions:
                c = cfg_at(cfg, res)
                for name, pc in sets.items():
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    base_mem = torch.cuda.memory_allocated()
                    m, s = cuda_time(lambda: imp._render(pc, cam(data, r_view), bg, c), a.reps, a.warmup)
                    peak = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20
                    rows.setdefault((name, res), {"ms": [], "std": [], "mem": [], "n": []})
                    rows[(name, res)]["ms"].append(m); rows[(name, res)]["std"].append(s)
                    rows[(name, res)]["mem"].append(peak); rows[(name, res)]["n"].append(pc["xyz"].shape[0])
                if res in a.seq_resolutions:
                    for name, pc in sets.items():
                        cams = [cam(data, r) for r in range(1, 1 + a.seq_frames)]
                        for cc in cams[:5]:
                            imp._render(pc, cc, bg, c)                   # warm-up
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        for cc in cams:
                            imp._render(pc, cc, bg, c)
                        torch.cuda.synchronize()
                        seq.setdefault((name, res), []).append(1000 * (time.perf_counter() - t0))
        if obj_i == a.video_obj:
            video_data = (data, sets)
        print(f"object {obj_i} done", flush=True)

    # ---- table
    out_rows = []
    for (name, res), v in rows.items():
        n = int(np.median(v["n"]))
        out_rows.append({
            "variant": name, "resolution": res, "gaussians": n,
            "storage_MB": round(n * BYTES_PER_GAUSSIAN / 1e6, 3),
            "frame_ms": round(float(np.mean(v["ms"])), 3), "frame_ms_std": round(float(np.mean(v["std"])), 3),
            "fps": round(1000 / float(np.mean(v["ms"])), 1),
            "peak_render_MB": round(float(np.mean(v["mem"])), 1),
            "seq_ms": round(float(np.mean(seq[(name, res)])), 1) if (name, res) in seq else "",
            "gpu": gpu})
    base_ms = {r["resolution"]: r["frame_ms"] for r in out_rows if r["variant"] == "Full model"}
    for r in out_rows:
        r["speedup_vs_full"] = round(base_ms[r["resolution"]] / r["frame_ms"], 2)
    with open(os.path.join(a.out, "cost_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    cols = ["variant", "resolution", "gaussians", "storage_MB", "frame_ms", "fps", "speedup_vs_full",
            "peak_render_MB", "seq_ms"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(out_rows, key=lambda r: (r["resolution"], -r["gaussians"])):
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md.append("")
    md.append(f"GPU: {gpu}. Stage times (mean over {a.n_timing} objects): "
              f"U-Net forward {np.mean(stage['unet_ms']):.2f} ms, hybrid selection {np.mean(stage['select_hybrid_ms']):.2f} ms, "
              f"4x4 merging {np.mean(stage['select_merge_ms']):.2f} ms. seq_ms = {a.seq_frames}-frame 360-degree sequence.")
    open(os.path.join(a.out, "cost_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)

    # ---- video + strip
    data, sets = video_data
    order = [v[0] for v in variants if v[0] in a.video_panels or not a.video_panels]
    c = cfg_at(cfg, a.video_res)
    frames_dir = os.path.join(a.out, f"frames_obj{a.video_obj}")
    os.makedirs(frames_dir, exist_ok=True)
    frames = []
    for r in range(1, data["gt_images"].shape[1]):
        panels = []
        for name in order:
            img = to_uint8(imp._render(sets[name], cam(data, r), bg, c))
            panels.append(label_panel(img, name, f"{sets[name]['xyz'].shape[0]:,} Gaussians"))
        frame = np.concatenate(panels, axis=1)
        frames.append(frame)
        Image.fromarray(frame).save(os.path.join(frames_dir, f"{r:04d}.png"))
    path = write_video(frames, os.path.join(a.out, f"video_obj{a.video_obj}"), a.fps)
    print("video:", path, flush=True)

    ang = torch.rad2deg(torch.acos((torch.nn.functional.normalize(data["camera_centers"][0], dim=-1)
                                    * torch.nn.functional.normalize(data["camera_centers"][0, :1], dim=-1)).sum(-1).clamp(-1, 1)))
    picks = [int(torch.argmin((ang[1:] - t).abs())) + 1 for t in a.strip_angles]
    rows_img = []
    for r in picks:
        panels = [label_panel(to_uint8(torch.nn.functional.interpolate(data["gt_images"][0, r][None], size=(a.video_res, a.video_res),
                                                                       mode="bilinear", align_corners=False)[0]),
                              "Ground truth", f"{ang[r]:.0f} deg from input")]
        for name in order:
            panels.append(label_panel(to_uint8(imp._render(sets[name], cam(data, r), bg, c)), name,
                                      f"{sets[name]['xyz'].shape[0]:,} Gaussians"))
        rows_img.append(np.concatenate(panels, axis=1))
    Image.fromarray(np.concatenate(rows_img, axis=0)).save(os.path.join(a.out, f"strip_obj{a.video_obj}.png"))
    print("strip:", os.path.join(a.out, f"strip_obj{a.video_obj}.png"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ft_path", default=None, help="Level-3 run dir, e.g. runs/ft10_hybrid")
    p.add_argument("--ft_ckpt", default="model_best.pth")
    p.add_argument("--ft_keep", type=float, default=0.10)
    p.add_argument("--n_timing", type=int, default=10, help="test objects used for timing")
    p.add_argument("--resolutions", nargs="+", type=int, default=[128, 256, 512])
    p.add_argument("--seq_resolutions", nargs="+", type=int, default=[128, 512])
    p.add_argument("--seq_frames", type=int, default=100)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--video_obj", type=int, default=0)
    p.add_argument("--video_res", type=int, default=256)
    p.add_argument("--video_panels", nargs="*", default=["Full model", "Opacity 10%", "Hybrid 10%",
                                                          "Fine-tuned + hybrid 10% (ours)"])
    p.add_argument("--strip_angles", nargs="+", type=float, default=[45, 100, 160])
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out", default="results/cost")
    main(p.parse_args())
