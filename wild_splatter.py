# entirely new code file from Taya
"""
"In the wild": run the car models on your own images (internet photos, renders, toy cars) and render
novel views + a 360-degree video, several models side by side.

Camera: Splatter Image needs the input camera. All SRN test cars share the same camera trajectory, so we
borrow the cameras of a TEMPLATE test car and replace its input photo with yours. Your car is cropped,
scaled and centred to occupy the same box as the template car, and put on a white background (128x128).

Input images: best a PNG WITH TRANSPARENT BACKGROUND (macOS: right-click the image in Finder ->
Quick Actions -> Remove Background). JPGs without alpha are used as they are (a plain white/light
background works; a busy background will not). The car should be seen from the side-front and slightly
from above, like the SRN input views (see <out>/template_input.png).

  python wild_splatter.py --images wild/*.png --out results/wild --angles 90 180 \\
    --model "Splatter Image|released|-|none|1.0" \\
    --model "Fine-tuned 10% + symmetry|runs/sym_ft10|model_best.pth|hybrid|0.1"
Model spec as in vis_splatter.py: "label|path or released|ckpt|mask|keep[|twins_off]".
Outputs per image <name>: <name>_input.png (what the model sees), <name>_views.png/.pdf
(rows = models, columns = input + views), <name>_360.mp4 (or .gif) with all models side by side.
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Subset

import importance as imp
from datasets.dataset_factory import get_dataset
from eval_budget import FreeCacheDataset, load_model, model_input
from vis_splatter import cfg_at, label_bar, parse_model, select, storage_mb, to_img


def fg_box(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def prepare(path, template_rgb, res, margin_thr=0.97):
    """Your photo -> (3, res, res) tensor in [0, 1], placed like the template car."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        im = im.convert("RGBA")
        rgba = np.asarray(im).astype(np.float32) / 255.0
        alpha = rgba[..., 3:]
        rgb = rgba[..., :3] * alpha + (1 - alpha)                 # composite on white
        mask = alpha[..., 0] > 0.5
    else:
        print(f"  {os.path.basename(path)}: no transparency, assuming a light background", flush=True)
        rgb = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
        mask = rgb.min(-1) < margin_thr
    x0, y0, x1, y1 = fg_box(mask)
    crop = Image.fromarray((rgb[y0:y1, x0:x1] * 255).astype(np.uint8))
    t = template_rgb.permute(1, 2, 0).cpu().numpy()                 # template input, (res, res, 3)
    tx0, ty0, tx1, ty1 = fg_box(t.min(-1) < margin_thr)
    s = min((tx1 - tx0) / crop.width, (ty1 - ty0) / crop.height)
    w, h = max(1, round(crop.width * s)), max(1, round(crop.height * s))
    crop = crop.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (res, res), (255, 255, 255))
    cx, cy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    canvas.paste(crop, (int(round(cx - w / 2)), int(round(cy - h / 2))))
    return torch.from_numpy(np.asarray(canvas).astype(np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def main(a):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    specs = [parse_model(s) for s in a.model]
    models = []
    for s in specs:
        m, cfg = load_model("cars", s["path"], device, s["ckpt"] or "model_latest.pth")
        models.append((s, m.eval(), cfg))
    cfg0 = models[0][2]
    bg = torch.tensor([1, 1, 1] if cfg0.data.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    rcfg = cfg_at(cfg0, a.res)

    test = get_dataset(cfg0, "test")
    data0 = next(iter(DataLoader(Subset(FreeCacheDataset(test), [a.template]), batch_size=1)))
    data0 = {k: v.to(device) for k, v in data0.items()}
    os.makedirs(a.out, exist_ok=True)
    Image.fromarray(to_img(data0["gt_images"][0, 0])).save(os.path.join(a.out, "template_input.png"))

    F = data0["gt_images"].shape[1]
    c = data0["camera_centers"][0]
    nrm = torch.nn.functional.normalize
    ang = torch.rad2deg(torch.acos((nrm(c, dim=-1) * nrm(c[:1], dim=-1)).sum(-1).clamp(-1, 1)))
    views = [int(torch.argmin((ang[1:] - t).abs())) + 1 for t in a.angles]
    cams = lambda r: (data0["world_view_transforms"][0, r], data0["full_proj_transforms"][0, r], data0["camera_centers"][0, r])

    for path in a.images:
        name = os.path.splitext(os.path.basename(path))[0]
        img = prepare(path, data0["gt_images"][0, 0], data0["gt_images"].shape[-1]).to(device)
        Image.fromarray(to_img(img)).save(os.path.join(a.out, f"{name}_input.png"))
        data = dict(data0)
        data["gt_images"] = data0["gt_images"].clone()
        data["gt_images"][0, 0] = img

        cols = []
        for s, m, cfg in models:
            kw = {"sym_switch_override": 0.0} if s["twins_off"] else {}
            recon = m(model_input(data, cfg), data["view_to_world_transforms"][:, :1], data["source_cv2wT_quat"][:, :1], None, **kw)
            pc, orig, kept, switch = select(recon, s["mask"], s["keep"])
            cols.append(dict(spec=s, pc=pc, n_stored=int(kept.sum()), sym=switch is not None))

        # rows = models, columns = input + requested views
        nr, nc = len(cols), 1 + len(views)
        fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.3 * nr), squeeze=False)
        for i, cl in enumerate(cols):
            axes[i, 0].imshow(to_img(img), interpolation="lanczos")   # model sees 128x128; smooth only for display
            mb = storage_mb(cl["n_stored"], cl["sym"])
            axes[i, 0].set_ylabel(f"{cl['spec']['label']}\n{mb:.2f} MB", fontsize=8)
            for j, v in enumerate(views):
                axes[i, j + 1].imshow(to_img(imp._render(cl["pc"], cams(v), bg, rcfg)))
                if i == 0:
                    axes[0, j + 1].set_title(f"{ang[v]:.0f}° from input", fontsize=9)
            for ax in axes[i]:
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
        axes[0, 0].set_title("Input (your photo)", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.out, f"{name}_views.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        if not a.no_video:
            frames = []
            inp_panel = label_bar(np.asarray(Image.fromarray(to_img(img)).resize((a.res, a.res))), "input", size=11)
            for r in range(1, F, a.video_step):
                panels = [inp_panel] + [label_bar(to_img(imp._render(cl["pc"], cams(r), bg, rcfg)), cl["spec"]["label"], size=11)
                                        for cl in cols]
                frames.append(np.concatenate(panels, axis=1))
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
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--template", type=int, default=85, help="test car whose cameras (and box) are borrowed")
    p.add_argument("--angles", nargs="+", type=float, default=[90, 180])
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--video_step", type=int, default=1)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--out", default="results/wild")
    main(p.parse_args())
