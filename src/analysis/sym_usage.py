# entirely new code file from Taya
"""
Ablation F: HOW does a symmetric model use its twins under a Gaussian budget?  (no rendering, fast)

For each model and each stored budget (hybrid rank on the ORIGINAL Gaussians) we report, over test objects:
  near_side_kept   fraction of the kept originals that lie on the input camera's side of the mirror plane
  near_side_all    the same fraction among all opaque originals (opacity > 0.05), for reference
  switch_kept      mean twin switch of the kept originals (1 = twin fully on)
  twins_on_kept    fraction of kept originals whose twin is clearly on (switch > 0.5)
Hypothesis: under a budget the model keeps mostly near-side Gaussians and relies on their twins for the
hidden side, i.e. each stored Gaussian "works twice".

Run from a checkout with Paul's model code (e.g. ~/dl4cv-project/splatter-sym):
  python -m src.analysis.sym_usage --model "sym_long|experiments_out/sym_long|model_latest.pth" \\
                      --model "Sym+AnyBudget|runs/sym_anybudget|model_best.pth" \\
                      --model "Sym+FT10|runs/sym_ft10|model_best.pth" --max_objects 200 --out results/sym_usage
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.reduction import importance as imp
from datasets.dataset_factory import get_dataset
from src.reduction.eval_budget import FreeCacheDataset, load_model, model_input
from scene.symmetry import near_side_mask


@torch.no_grad()
def main(a):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    models = []
    for spec in a.model:
        label, path, ckpt = [x.strip() for x in spec.split("|")]
        m, cfg = load_model("cars", path, device, ckpt)
        models.append((label, m.eval(), cfg))
    cfg0 = models[0][2]
    test = get_dataset(cfg0, "test")
    n = len(test) if a.max_objects <= 0 else min(a.max_objects, len(test))
    loader = DataLoader(Subset(FreeCacheDataset(test), list(range(n))), batch_size=1, num_workers=a.num_workers)

    rows = []
    for obj, data in enumerate(loader):
        data = {k: v.to(device) for k, v in data.items()}
        for label, m, cfg in models:
            recon = m(model_input(data, cfg), data["view_to_world_transforms"][:, :1], data["source_cv2wT_quat"][:, :1], None)
            assert "sym_switch" in recon, f"{label} is not a symmetry model"
            plane, switch = recon["sym_plane"], recon["sym_switch"][0, :, 0]
            g, n_orig, _ = imp.split_symmetric(recon)
            orig = {k: v[0, :n_orig] for k, v in g.items()}
            near = near_side_mask(orig["xyz"][None], plane)[0, :, 0].bool()
            opaque = orig["opacity"][:, 0] > 0.05
            score = imp.cheap_scores(orig, "hybrid")
            for b in a.budgets:
                k = n_orig if b >= 1.0 else max(1, int(round(b * n_orig)))
                kept = imp.topk_indices(score, k) if b < 1.0 else opaque.nonzero()[:, 0]
                rows.append({"model": label, "budget": b, "object_idx": obj,
                             "near_side_kept": near[kept].float().mean().item(),
                             "near_side_all": near[opaque].float().mean().item() if opaque.any() else float("nan"),
                             "switch_kept": switch[kept].mean().item(),
                             "twins_on_kept": (switch[kept] > 0.5).float().mean().item()})
        if obj % 20 == 0:
            print(f"object {obj}/{n}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out + "_objects.csv", index=False)
    t = df.groupby(["model", "budget"])[["near_side_kept", "near_side_all", "switch_kept", "twins_on_kept"]].mean().reset_index()
    t.to_csv(a.out + "_summary.csv", index=False)
    print(t.round(3).to_string(index=False))
    print("(budget 1.0 = all opaque originals)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help='"label|run dir|checkpoint file"')
    p.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.1, 0.2, 1.0])
    p.add_argument("--max_objects", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out", default="results/sym_usage")
    main(p.parse_args())
