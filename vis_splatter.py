# entirely new code file from Taya
"""
"Figure 7"-style analysis (Splatter Image paper, Fig. 7: Input / Opacity / Location / Novel view) for
several models side by side, plus frames for a side-by-side 360-degree video.

Rows of the figure (one figure per object):
  Input       - the conditioning photo (same for all columns)
  Opacity     - the 128x128 opacity map of the predicted (original) Gaussians; pixels whose Gaussian was
                DISCARDED by the budget mask are shown in red, so the kept 10 % are visible
  Location    - xyz of each pixel's Gaussian as RGB (same colour scale for all columns of the object);
                discarded pixels in grey
  Twin switch - symmetric models only: how strongly each pixel's mirrored twin is switched on
  Views       - renders at the requested input-target angles (closest test camera), and the ground truth

Run from a checkout that contains Paul's model code (it also loads plain models), e.g.
~/dl4cv-project/splatter-sym on Taya's account:

  python vis_splatter.py --objects 0 57 312 --out results/vis \\
    --model "Original (all 16,384)|released|-|none|1.0" \\
    --model "Original + hybrid 10%|released|-|hybrid|0.1" \\
    --model "Fine-tuned for 10% (ours)|$HOME/dl4cv-project/splatter-image/runs/ft10_hybrid|model_best.pth|hybrid|0.1" \\
    --model "Fine-tuned 10% + symmetry|runs/sym_ft10|model_best.pth|hybrid|0.1" \\
    --video_obj 0
Model spec: "label|path or 'released'|checkpoint file|mask (none/hybrid/opacity)|kept fraction[|twins_off]".
Also writes teaser_obj<k>_view<v>.png: input, one novel view per model (MB above, PSNR below), ground truth.
Outputs in --out: fig7_obj<k>.png/.pdf, compare_obj<k>.png/.pdf (views x models, PSNR per render), panels/<k>/..., video_obj<k>/frames/*.png (+ .mp4 or .gif).
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

import importance as imp
from datasets.dataset_factory import get_dataset
from eval_budget import FreeCacheDataset, load_model, model_input

H = W = 128
DISCARD_RGB = np.array([0.93, 0.42, 0.42])    # discarded pixels in the opacity row
GREY_RGB = np.array([0.80, 0.80, 0.80])       # discarded pixels in the location row


def cfg_at(cfg, res):
    c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    c.data.training_resolution = int(res)
    return c


def parse_model(s):
    parts = [x.strip() for x in s.split("|")]
    label, path, ckpt, mask, keep = parts[:5]
    opt = parts[5] if len(parts) > 5 else ""          # optional: "twins_off" (symmetry models only)
    return dict(label=label, path=None if path == "released" else os.path.expanduser(path),
                ckpt=None if ckpt == "-" else ckpt, mask=mask, keep=float(keep), twins_off=(opt == "twins_off"))


def storage_mb(n_stored, sym):
    return n_stored * (27 if sym else 23) * 4 / 1e6


def to_img(t):
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def select(recon_b, mask, keep):
    """recon_b: batched model output for ONE object. Returns (render dict, kept-originals bool map (N,),
    switch (N,) or None)."""
    switch = recon_b["sym_switch"][0, :, 0].detach() if "sym_switch" in recon_b else None
    g, n_orig, sym = imp.split_symmetric(recon_b)
    pc = {k: v[0].contiguous() for k, v in g.items()}
    orig = {k: v[:n_orig] for k, v in pc.items()}
    kept = torch.ones(n_orig, dtype=torch.bool, device=pc["xyz"].device)
    if mask != "none" and keep < 1.0:
        k = max(1, int(round(keep * n_orig)))
        idx = imp.topk_indices(imp.cheap_scores(orig, mask), k)
        kept = torch.zeros_like(kept); kept[idx] = True
        pc = imp.subset(pc, imp.pair_indices(idx, n_orig, sym))
    return pc, orig, kept, switch


def label_bar(arr, text, size=14):
    h, w, _ = arr.shape
    bar = Image.new("RGB", (w, 26), (255, 255, 255))
    d = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    d.text((4, 4), text, fill=(20, 20, 20), font=font)
    return np.concatenate([arr, np.asarray(bar)], axis=0)


@torch.no_grad()
def main(a):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    specs = [parse_model(s) for s in a.model]
    models = []
    base_cfg = None
    for s in specs:
        m, cfg = load_model("cars", s["path"], device, s["ckpt"] or "model_latest.pth")
        models.append((s, m.eval(), cfg))
        if base_cfg is None:
            base_cfg = cfg
    bg = torch.tensor([1, 1, 1] if base_cfg.data.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    rcfg = cfg_at(base_cfg, a.res)

    test = get_dataset(base_cfg, "test")
    objs = sorted(set(a.objects + a.video_obj))
    loader = DataLoader(Subset(FreeCacheDataset(test), objs), batch_size=1, num_workers=1)
    os.makedirs(a.out, exist_ok=True)

    for obj, data in zip(objs, loader):
        data = {k: v.to(device) for k, v in data.items()}
        F = data["gt_images"].shape[1]
        c = data["camera_centers"][0]
        cos = (torch.nn.functional.normalize(c, dim=-1) * torch.nn.functional.normalize(c[:1], dim=-1)).sum(-1).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(cos))
        views = [int(torch.argmin((ang[1:] - t).abs())) + 1 for t in a.angles]
        if a.views:                                   # explicit test-camera indices (view_id of eval_budget CSVs)
            views = a.views
        inp = model_input(data, base_cfg)
        cams = lambda r: (data["world_view_transforms"][0, r], data["full_proj_transforms"][0, r], data["camera_centers"][0, r])

        cols = []
        for s, m, cfg in models:
            kw = {"sym_switch_override": 0.0} if s["twins_off"] else {}
            recon = m(model_input(data, cfg), data["view_to_world_transforms"][:, :1], data["source_cv2wT_quat"][:, :1], None, **kw)
            pc, orig, kept, switch = select(recon, s["mask"], s["keep"])
            cols.append(dict(spec=s, pc=pc, orig=orig, kept=kept.cpu().numpy(), switch=None if switch is None else switch.cpu().numpy(),
                             n_stored=int(kept.sum()), sym=switch is not None))

        # common location colour scale: robust bounds of the opaque Gaussians of the FIRST column
        ref = cols[0]["orig"]
        opq = ref["opacity"][:, 0] > 0.05
        xyz_ref = ref["xyz"][opq if opq.sum() > 100 else slice(None)].cpu().numpy()
        lo, hi = np.percentile(xyz_ref, 1, axis=0), np.percentile(xyz_ref, 99, axis=0)

        has_switch = any(cl["sym"] for cl in cols)
        row_names = ["Input", "Opacity (red = discarded)", "Location (xyz as RGB)"] + (["Twin switch"] if has_switch else []) \
            + [f"View {ang[v]:.0f}° from input" for v in views]
        ncol = len(cols) + 1                       # + ground-truth column for the view rows
        fig, axes = plt.subplots(len(row_names), ncol, figsize=(2.1 * ncol, 2.1 * len(row_names)))
        inp_img = to_img(inp[0, 0, :3])
        for j in range(ncol):
            for i in range(len(row_names)):
                axes[i, j].axis("off")
        for j, cl in enumerate(cols):
            kept = cl["kept"].reshape(H, W)
            op = cl["orig"]["opacity"][:, 0].cpu().numpy().reshape(H, W)
            op_rgb = np.repeat(op[..., None], 3, axis=2)
            op_rgb[~kept] = DISCARD_RGB
            loc = ((cl["orig"]["xyz"].cpu().numpy() - lo) / np.maximum(hi - lo, 1e-6)).clip(0, 1).reshape(H, W, 3)
            loc[~kept] = GREY_RGB
            r = 0
            axes[r, j].imshow(inp_img); r += 1
            axes[r, j].imshow(op_rgb); r += 1
            axes[r, j].imshow(loc); r += 1
            if has_switch:
                if cl["sym"]:
                    sw = cl["switch"].reshape(H, W) * kept * (0.0 if cl["spec"]["twins_off"] else 1.0)
                    axes[r, j].imshow(sw, cmap="magma", vmin=0, vmax=1)
                else:
                    axes[r, j].text(0.5, 0.5, "no symmetry", ha="center", va="center", fontsize=8, color="0.5",
                                    transform=axes[r, j].transAxes)
                r += 1
            for v in views:
                axes[r, j].imshow(to_img(imp._render(cl["pc"], cams(v), bg, rcfg))); r += 1
            sub = f"{cl['n_stored']:,} stored" + ((" (twins off)" if cl["spec"]["twins_off"] else " (+ twins)") if cl["sym"] else "")
            axes[0, j].set_title(f"{cl['spec']['label']}\n{sub}", fontsize=8)
        # ground-truth column (views only)
        axes[0, -1].set_title("Ground truth", fontsize=8)
        r0 = 4 if has_switch else 3
        for k, v in enumerate(views):
            gt = torch.nn.functional.interpolate(data["gt_images"][0, v][None], size=(a.res, a.res), mode="bilinear", align_corners=False)[0]
            axes[r0 + k, -1].imshow(to_img(gt))
        for i, name in enumerate(row_names):
            axes[i, 0].text(-0.08, 0.5, name, rotation=90, ha="right", va="center", fontsize=8, transform=axes[i, 0].transAxes)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.out, f"fig7_obj{obj}.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"object {obj}: fig7 written", flush=True)

        # compare: rows = views, columns = models + ground truth, PSNR (128x128, as in evaluation) under each render
        bcfg = cfg_at(base_cfg, H)
        fig, axes = plt.subplots(len(views), len(cols) + 1, figsize=(2.2 * (len(cols) + 1), 2.35 * len(views)), squeeze=False)
        for i, v in enumerate(views):
            gt128 = data["gt_images"][0, v]
            for j, cl in enumerate(cols):
                axes[i, j].imshow(to_img(imp._render(cl["pc"], cams(v), bg, rcfg)))
                r128 = imp._render(cl["pc"], cams(v), bg, bcfg).clamp(0, 1)
                psnr = -10 * torch.log10(((r128 - gt128) ** 2).mean()).item()
                axes[i, j].set_xlabel(f"{psnr:.2f} dB", fontsize=9)
                if i == 0:
                    mb = storage_mb(cl["n_stored"], cl["sym"])
                    axes[0, j].set_title(f"{cl['spec']['label']}\n{mb:.2f} MB", fontsize=9)
            gt = torch.nn.functional.interpolate(gt128[None], size=(a.res, a.res), mode="bilinear", align_corners=False)[0]
            axes[i, -1].imshow(to_img(gt))
            axes[i, 0].set_ylabel(f"{ang[v]:.0f}° from input", fontsize=9)
        axes[0, -1].set_title("Ground truth", fontsize=9)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.out, f"compare_obj{obj}.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # teaser: input + one novel view per model + ground truth; MB on top, PSNR (128x128) below. One file per view.
        bcfg = cfg_at(base_cfg, H)
        for v0 in views:
            gt128 = data["gt_images"][0, v0]
            n = len(cols) + 2
            fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.8))
            axes[0].imshow(inp_img, interpolation="lanczos"); axes[0].set_title("Input photo", fontsize=9)
            for jj, cl in enumerate(cols):
                axes[jj + 1].imshow(to_img(imp._render(cl["pc"], cams(v0), bg, rcfg)))
                r128 = imp._render(cl["pc"], cams(v0), bg, bcfg).clamp(0, 1)
                psnr = -10 * torch.log10(((r128 - gt128) ** 2).mean()).item()
                mb = storage_mb(cl["n_stored"], cl["sym"])
                axes[jj + 1].set_title(f"{cl['spec']['label']}\n{mb:.2f} MB", fontsize=9)
                axes[jj + 1].set_xlabel(f"PSNR {psnr:.2f} dB", fontsize=9)
            gt = torch.nn.functional.interpolate(gt128[None], size=(a.res, a.res), mode="bilinear", align_corners=False)[0]
            axes[-1].imshow(to_img(gt)); axes[-1].set_title(f"Ground truth\n{ang[v0]:.0f}° from input", fontsize=9)
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(a.out, f"teaser_obj{obj}_view{v0}.{ext}"), dpi=200, bbox_inches="tight")
            plt.close(fig)

        if obj in a.video_obj:
            fdir = os.path.join(a.out, f"video_obj{obj}", "frames")
            os.makedirs(fdir, exist_ok=True)
            frames = []
            for i, r in enumerate(range(1, F, a.video_step)):
                panels = []
                for cl in cols:
                    img = to_img(imp._render(cl["pc"], cams(r), bg, rcfg))
                    panels.append(label_bar(img, f"{cl['spec']['label']}  ({cl['n_stored']:,})", size=11))
                frame = np.concatenate(panels, axis=1)
                frames.append(frame)
                Image.fromarray(frame).save(os.path.join(fdir, f"{i:04d}.png"))
            try:
                import imageio
                imageio.mimsave(os.path.join(a.out, f"video_obj{obj}.mp4"), frames, fps=a.fps, quality=8, macro_block_size=1)
                print("video: mp4 written", flush=True)
            except Exception as ex:
                ims = [Image.fromarray(f) for f in frames]
                ims[0].save(os.path.join(a.out, f"video_obj{obj}.gif"), save_all=True, append_images=ims[1:],
                            duration=int(1000 / a.fps), loop=0)
                print(f"video: mp4 failed ({ex.__class__.__name__}), GIF written; frames in {fdir}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help='"label|path or released|ckpt|mask|keep"')
    p.add_argument("--objects", nargs="+", type=int, default=[0])
    p.add_argument("--angles", nargs="+", type=float, default=[100, 160], help="input-target angles of the view rows")
    p.add_argument("--views", nargs="+", type=int, default=None, help="explicit test-camera indices (overrides --angles)")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--video_obj", nargs="+", type=int, default=[], help="objects to make a 360-degree video for")
    p.add_argument("--video_step", type=int, default=1, help="use every k-th test camera for the video")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--out", default="results/vis")
    main(p.parse_args())
