# entirely new code file from Taya
"""
Any object "in the wild" (bottle, mug, toy, ...): the official multi-category Splatter Image model trained on
Objaverse (the one behind the authors' online demo), plus our TRAINING-FREE budget selection (hybrid rank)
applied to it at test time. Our fine-tuned / symmetry models exist only for cars, so they are not used here.

Pre-processing and the 360-degree camera loop are the ones of the official demo (gradio_app.py):
object cropped, padded to a square, foreground ratio 0.65, white background, 128x128.

Input: a PNG WITH TRANSPARENT BACKGROUND (macOS Finder: right-click -> Quick Actions -> Remove Background).
A JPG with a plain white/light background also works (the light pixels become the background).

  python -m src.visualization.wild_objaverse --images wild_any/*.png --budgets 1.0 0.1 0.05 --out results/wild_any
Outputs per image <name>: <name>_input.png, <name>_views.png/.pdf (rows = budgets, columns = input +
views at --angles), <name>_360.mp4 (all budgets side by side).
"""
import argparse
import os
import sys
import types

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from huggingface_hub import hf_hub_download

for _m in ("rembg", "plyfile"):            # app_utils imports these at top level; we only need its cameras
    try:
        __import__(_m)
    except ImportError:
        stub = types.ModuleType(_m)
        stub.PlyData = stub.PlyElement = None
        sys.modules[_m] = stub
from utils.app_utils import get_source_camera_v2w_rmo_and_quats, get_target_cameras  # noqa: E402
from scene.gaussian_predictor import GaussianSplatPredictor  # noqa: E402

from src.reduction import importance as imp  # noqa: E402
from src.visualization.vis_splatter import cfg_at, label_bar, to_img  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_LOOP = 200                                # frames of the official 360-degree loop


def prepare(path, ratio=0.65, thr=0.97):
    """Same steps as the official demo: crop the object, pad to a square, foreground ratio, white bg, 128."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        rgba = np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0
    else:
        print(f"  {os.path.basename(path)}: no transparency, assuming a light background", flush=True)
        rgb = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
        rgba = np.concatenate([rgb, (rgb.min(-1, keepdims=True) < thr).astype(np.float32)], -1)
    ys, xs = np.nonzero(rgba[..., 3] > 0.5)
    fg = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rgb = fg[..., :3] * fg[..., 3:] + (1 - fg[..., 3:])
    h, w = rgb.shape[:2]
    size = int(max(h, w) / ratio)
    canvas = np.ones((size, size, 3), np.float32)
    canvas[(size - h) // 2:(size - h) // 2 + h, (size - w) // 2:(size - w) // 2 + w] = rgb
    img = Image.fromarray((canvas * 255).astype(np.uint8)).resize((128, 128), Image.LANCZOS)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)


def keep(pc, b):
    if b >= 1.0:
        return pc, pc["xyz"].shape[0]
    k = max(1, int(round(b * pc["xyz"].shape[0])))
    return imp.subset(pc, imp.topk_indices(imp.cheap_scores(pc, "hybrid"), k)), k


@torch.no_grad()
def main(a):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "gradio_config.yaml"))
    model = GaussianSplatPredictor(cfg)
    ck = torch.load(hf_hub_download(repo_id="szymanowiczs/splatter-image-multi-category-v1", filename="model_latest.pth"),
                    map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    rcfg = cfg_at(cfg, a.res)
    bg = torch.ones(3, device=device)

    v2w, quats = get_source_camera_v2w_rmo_and_quats()
    wvt, fpt, cc = get_target_cameras()
    cams = lambda r: (wvt[r].to(device), fpt[r].to(device), cc[r].to(device))
    views = [int(round(t / 360 * N_LOOP)) % N_LOOP for t in a.angles]   # the loop is evenly spaced in azimuth
    os.makedirs(a.out, exist_ok=True)

    for path in a.images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = prepare(path).to(device)
        Image.fromarray(to_img(img)).save(os.path.join(a.out, f"{name}_input.png"))
        recon = model(img[None, None], v2w.to(device), quats.to(device), None)
        pc = {k: v[0].contiguous() for k, v in recon.items() if torch.is_tensor(v) and v.dim() >= 3}
        rows = []
        for b in a.budgets:
            sel, n = keep(pc, b)
            rows.append((f"{'all' if b >= 1 else f'{b * 100:g}%'} ({n:,} Gaussians, {n * 23 * 4 / 1e6:.2f} MB)", sel))

        fig, axes = plt.subplots(len(rows), 1 + len(views), figsize=(2.3 * (1 + len(views)), 2.3 * len(rows)), squeeze=False)
        for i, (lab, sel) in enumerate(rows):
            axes[i, 0].imshow(to_img(img))
            axes[i, 0].set_ylabel(lab.replace(" (", "\n("), fontsize=8)
            for j, v in enumerate(views):
                axes[i, j + 1].imshow(to_img(imp._render(sel, cams(v), bg, rcfg)))
            for ax in axes[i]:
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
        axes[0, 0].set_title("Input", fontsize=9)
        for j, t in enumerate(a.angles):
            axes[0, j + 1].set_title(f"{t:g}° around", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.out, f"{name}_views.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        if not a.no_video:
            frames = []
            for r in range(0, N_LOOP, a.video_step):
                frames.append(np.concatenate([label_bar(to_img(imp._render(sel, cams(r), bg, rcfg)), lab.split(" (")[0], size=11)
                                              for lab, sel in rows], axis=1))
            try:
                import imageio
                imageio.mimsave(os.path.join(a.out, f"{name}_360.mp4"), frames, fps=a.fps, quality=8, macro_block_size=1)
            except Exception as ex:
                ims = [Image.fromarray(f) for f in frames]
                ims[0].save(os.path.join(a.out, f"{name}_360.gif"), save_all=True, append_images=ims[1:],
                            duration=int(1000 / a.fps), loop=0)
                print(f"  mp4 failed ({ex.__class__.__name__}), GIF written", flush=True)
        print(f"{name}: done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--budgets", nargs="+", type=float, default=[1.0, 0.1, 0.05])
    p.add_argument("--angles", nargs="+", type=float, default=[90, 180])
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--video_step", type=int, default=1)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--out", default="results/wild_any")
    main(p.parse_args())
