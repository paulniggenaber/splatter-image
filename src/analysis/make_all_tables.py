# entirely new code file from Taya
"""
Recompute EVERY number of the report from the raw CSV files, in one protocol, with sources.

    python -m src.analysis.make_all_tables --results results --out results/final_numbers

Protocol used everywhere (unless a table says otherwise):
  * SRN Cars test set, 703 objects: blank objects (car 375, all 251 images white) are dropped
    automatically (median masked PSNR of the original model > 30 dB);
  * 25 evaluation views per object (odd novel views); even views are used only by oracle scores;
  * per-object means first, then mean over objects; "delta" columns are PAIRED per-object
    differences with a 95 % confidence interval (1.96 * std / sqrt(n)).
Exceptions, clearly marked: 01 (official eval.py, all 250 views) and 02a (all 250 views).

Inputs expected in --results (missing ones are skipped with a note):
  budget_full_part*_{objects,views}.csv      Level 1, original model (both runs, incl. _new*)
  test_head_A_*, test_abl_*_*                 Level 2 heads
  test_ft_ctrl_*, test_ft10_learned_*, test_ft10_hybrid_*, test_ft5_hybrid_*, test_ft_anybudget_*   Level 3
  symA_full_*, test_sym_ft10_*, test_sym_anybudget_*                                               symmetry x budget
  cost/cost_table.csv                         cost benchmark
  raw_official/baseline_cars_full_scores.txt, raw_official/viewpoint_gap_full704.csv
  head_logs/<run>/log.csv                     Level 2 validation overlap (optional, from the cluster)
  train_logs/<run>.out                        fine-tuning logs, for validation curves (optional)
Every output table lists its source files.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from src.analysis import analyze_level1 as A  # noqa: E402  (load(), ci95(), bin_angles())

BINS = list(range(0, 181, 30))
BIN_LABELS = ["30-60", "60-90", "90-120", "120-150", "150-180"]
METRICS = ["psnr", "psnr_masked", "ssim", "lpips"]
FLOATS_PER_GAUSSIAN = {"plain": 23, "symmetric": 27}   # sym: + twin switch (1) + twin colour correction (3)

# model name -> (file prefix, stored-Gaussian kind)
RUNS = {
    "original":         ("budget_full_part",   "plain"),
    "head_A":           ("test_head_A",        "plain"),
    "control":          ("test_ft_ctrl",       "plain"),
    "FT10-learned":     ("test_ft10_learned",  "plain"),
    "FT10-hybrid":      ("test_ft10_hybrid",   "plain"),
    "FT5-hybrid":       ("test_ft5_hybrid",    "plain"),
    "AnyBudget":        ("test_ft_anybudget",  "plain"),
    "Sym (post-hoc)":   ("symA_full",          "symmetric"),
    "Sym+FT10":         ("test_sym_ft10",      "symmetric"),
    "Sym+AnyBudget":    ("test_sym_anybudget", "symmetric"),
}
ABLATIONS = ["abl_reference", "abl_inputs_gauss", "abl_inputs_rgb", "abl_loss_mse", "abl_target_errsens"]


# ----------------------------------------------------------------------------- helpers
class Out:
    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.index = []

    def write(self, name, title, blocks, sources):
        path = os.path.join(self.folder, name + ".md")
        with open(path, "w") as f:
            f.write(f"# {title}\n\n")
            for b in blocks:
                f.write(b.rstrip() + "\n\n")
            f.write("**Sources:** " + ", ".join(f"`{s}`" for s in sources) + "\n")
        self.index.append((name, title))
        print("wrote", path)

    def csv(self, name, df):
        df.to_csv(os.path.join(self.folder, name + ".csv"), index=False)


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("—" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def files_for(res, prefix):
    return sorted(glob.glob(os.path.join(res, f"{prefix}*_views.csv")))


def load_views(res, prefix, model):
    fs = files_for(res, prefix)
    if not fs:
        return None
    v = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    v = v.drop_duplicates(["object_idx", "criterion", "budget", "view_id"])
    return v.assign(model=model)


def pm(d):
    return f"{d.mean():+.3f} ± {A.ci95(d):.3f}"


# ----------------------------------------------------------------------------- main
def main(a):
    res, out = a.results, Out(a.out)

    # ---------- load all test-set evaluations (25-view protocol)
    V, sources = [], {}
    for model, (prefix, _) in RUNS.items():
        v = load_views(res, prefix, model)
        if v is None:
            print(f"note: no files for {model} ({prefix}*) - skipped")
            continue
        if model == "original":
            # Level 1 also contains criteria that other runs share names with; keep all
            pass
        V.append(v)
        sources[model] = [os.path.basename(f) for f in files_for(res, prefix)]
    for abl in ABLATIONS:
        v = load_views(res, f"test_{abl}", f"head:{abl}")
        if v is not None:
            V.append(v[v.criterion == "learned"])
            sources[f"head:{abl}"] = [os.path.basename(f) for f in files_for(res, f"test_{abl}")]
    V = pd.concat(V, ignore_index=True)

    orig_full = V[(V.model == "original") & (V.criterion == "full")]
    med = orig_full.groupby("object_idx").psnr_masked.median()
    blank = med[med > 30].index.tolist()
    V = V[~V.object_idx.isin(blank)]
    empty = orig_full[orig_full.psnr_masked >= 99.9][["object_idx", "view_id"]].assign(_e=1)
    V = V.merge(empty, on=["object_idx", "view_id"], how="left")
    V = V[V._e.isna()].drop(columns="_e")
    V["bin"] = A.bin_angles(V.angle_deg.astype(float), BINS)

    O = V.groupby(["model", "criterion", "budget", "object_idx"])[METRICS].mean().reset_index()
    ref = O[(O.model == "original") & (O.criterion == "full")].set_index("object_idx")
    n_obj = O.groupby("model").object_idx.nunique()

    def get(model, crit, budget):
        c = "full" if budget == 1.0 else crit
        g = O[(O.model == model) & (O.criterion == c) & np.isclose(O.budget, budget)]
        return g.set_index("object_idx") if len(g) else None

    def row(model, crit, budget, base=None):
        g = get(model, crit, budget)
        if g is None:
            return None
        r = {"model": model, "criterion": "full" if budget == 1.0 else crit, "budget": budget, "n_objects": len(g)}
        for m in METRICS:
            r[m] = g[m].mean()
        d = g.psnr - ref.loc[g.index, "psnr"]
        r["dPSNR_vs_original_full"] = pm(d)
        if base is not None:
            b = get(*base)
            if b is not None:
                common = g.index.intersection(b.index)
                r[f"dPSNR_vs_{base[0]}:{base[1]}@{base[2]:g}"] = pm(g.loc[common].psnr - b.loc[common].psnr)
        return r

    # ---------- 00 protocol
    blocks = [f"- Test objects after removing blank objects: **{ref.shape[0]}** (dropped: {blank}).",
              "- Views: 25 odd novel views per object (evenly spaced); oracle scores use the even views only.",
              "- Objects per evaluated model: " + ", ".join(f"{k}: {v}" for k, v in n_obj.items()),
              "- Deltas are paired per object, ± 95 % CI. Absolute numbers of the official protocol (all 250 views, "
              "tables 01 and 02a, and Paul's report) are NOT directly comparable with this protocol; differences are."]
    out.write("00_protocol", "Evaluation protocol", blocks, ["all files below"])

    # ---------- 01 reproduction (official protocol)
    bf = os.path.join(res, "raw_official", "baseline_cars_full_scores.txt")
    if os.path.exists(bf):
        b = pd.read_csv(bf, sep=r"\s+", header=None, names=["object_idx", "psnr", "ssim", "lpips"])
        t = pd.DataFrame([{"objects": "all 704", "psnr": b.psnr.mean(), "ssim": b.ssim.mean(), "lpips": b.lpips.mean()},
                          {"objects": "703 (without blank car 375)", **b[~b.object_idx.isin(blank)][["psnr", "ssim", "lpips"]].mean().to_dict()}])
        out.write("01_reproduction", "Reproduction of Splatter Image (official eval.py, all 250 views)",
                  [md(t, "{:.4f}"), "Paper (Table, SRN Cars): PSNR 24.00, SSIM 0.92, LPIPS 0.078."], [bf])

    # ---------- 02 viewpoint analysis of the full model
    blocks, srcs = [], []
    vf = os.path.join(res, "raw_official", "viewpoint_gap_full704.csv")
    if os.path.exists(vf):
        g = pd.read_csv(vf)
        g = g[~g.object_id.isin(blank)]
        g["bin"] = A.bin_angles(g.angle_deg, BINS)
        t = g.groupby("bin").agg(views=("psnr", "size"), psnr=("psnr", "mean"), ssim=("ssim", "mean"), lpips=("lpips", "mean"))
        t = t.loc[[x for x in BIN_LABELS if x in t.index]].reset_index()
        blocks += ["## a) Official protocol, all 250 views (703 objects)", md(t)]
        srcs.append(vf)
    f = V[(V.model == "original") & (V.criterion == "full")]
    t = f.groupby("bin").agg(views=("psnr", "size"), psnr=("psnr", "mean"), lpips=("lpips", "mean")).loc[BIN_LABELS].reset_index()
    blocks += ["## b) Our protocol, 25 views (703 objects)", md(t)]
    srcs += sources.get("original", [])
    out.write("02_viewpoint_full_model", "Quality of the full model vs input-target angle", blocks, srcs)

    # ---------- 03 Level 1
    crits = ["random", "opacity", "scale", "opacity_area", "hybrid", "edge_opacity", "contribution", "contribution_bal",
             "contribution_iter", "contribution_iter_bal", "error_sens", "error_sens_signed"]
    rows = [row("original", "full", 1.0)]
    for c in crits:
        for bgt in [0.25, 0.20, 0.10, 0.05, 0.02, 0.01]:
            r = row("original", c, bgt)
            if r:
                rows.append(r)
    for bgt in [0.25, 0.0625]:
        r = row("original", "merge", bgt)
        if r:
            rows.append(r)
    L1 = pd.DataFrame(rows)
    sel = V[(V.model == "original")].groupby(["criterion", "budget"]).size()   # noqa: F841 (existence check)
    # selection time from the objects files
    objs = pd.concat([pd.read_csv(x) for x in glob.glob(os.path.join(res, "budget_full_part*_objects.csv"))])
    st = objs[objs.object_idx > 0].groupby(["criterion", "budget"]).select_ms.mean().rename("select_ms").reset_index()
    L1 = L1.merge(st, on=["criterion", "budget"], how="left")
    out.csv("03_level1", L1)
    # angle table at 10 %
    f0 = V[(V.model == "original") & (V.criterion == "full")][["object_idx", "view_id", "psnr"]].rename(columns={"psnr": "p0"})
    W = V.merge(f0, on=["object_idx", "view_id"])
    W["d"] = W.psnr - W.p0
    ang = W[(W.model == "original") & np.isclose(W.budget, 0.10)].groupby(["criterion", "bin"]).d.mean().unstack("bin")
    ang = ang[[b for b in BIN_LABELS if b in ang.columns]]
    ang["worst"] = ang.min(axis=1)
    ang = ang.reset_index()
    out.write("03_level1", "Level 1 - training-free selection (original model)",
              ["Columns: mean PSNR / masked PSNR / SSIM / LPIPS over objects; dPSNR = paired difference to the full model.",
               md(L1[["criterion", "budget", "n_objects", "psnr", "dPSNR_vs_original_full", "psnr_masked", "ssim", "lpips", "select_ms"]]),
               "## ΔPSNR vs the full model per angle bin, 10 % kept (worst = smallest)", md(ang, "{:+.2f}")],
              sources.get("original", []))

    # ---------- 04 Level 2
    rows = [row("original", "opacity", 0.10), row("original", "opacity_area", 0.10), row("original", "hybrid", 0.10),
            row("original", "contribution", 0.10)]
    for bgt in [0.10, 0.05, 0.01]:
        r = row("head_A", "learned", bgt, base=("original", "contribution", bgt))
        if r:
            rows.append(r)
    for abl in ABLATIONS:
        r = row(f"head:{abl}", "learned", 0.10)
        if r:
            rows.append(r)
    L2 = pd.DataFrame([r for r in rows if r])
    blocks = [md(L2.drop(columns=[c for c in L2.columns if c in ("psnr_masked",)]))]
    logs = sorted(glob.glob(os.path.join(res, "head_logs", "*", "log.csv")))
    if logs:
        vals = []
        for lp in logs:
            lg = pd.read_csv(lp)
            last = lg.iloc[-1]
            vals.append({"run": os.path.basename(os.path.dirname(lp)), "step": int(last.step),
                         "val_overlap10": last.val_overlap10, "val_overlap5": last.val_overlap5,
                         "opacity_overlap10": last.opacity_overlap10})
        blocks += ["## Validation overlap with the oracle's top-10 % (last logged step)", md(pd.DataFrame(vals))]
    else:
        blocks += ["(Validation overlaps: copy `runs/<run>/log.csv` from the cluster to `results/head_logs/<run>/log.csv`.)"]
    out.csv("04_level2", L2)
    out.write("04_level2", "Level 2 - learned importance head", blocks,
              sum([sources.get(k, []) for k in sources if k.startswith("head")], []) + [os.path.relpath(x, res) for x in logs])

    # ---------- 05 Level 3
    plan = [("original", "hybrid"), ("control", "hybrid"), ("FT10-learned", "learned"), ("FT10-hybrid", "hybrid"),
            ("FT5-hybrid", "hybrid"), ("AnyBudget", "hybrid")]
    rows = []
    for m, c in plan:
        for bgt in [0.02, 0.05, 0.10, 0.20, 0.25, 1.0]:
            r = row(m, c, bgt, base=None if m == "original" else ("original", "hybrid", bgt))
            if r:
                rows.append(r)
    L3 = pd.DataFrame(rows)
    piv = L3.pivot_table(index="model", columns="budget", values="psnr").reindex([p[0] for p in plan])
    piv.columns = [("100%" if c == 1.0 else f"{c * 100:g}%") for c in piv.columns]
    ang3 = W[W.model.isin([p[0] for p in plan]) & (W.criterion.isin(["hybrid", "learned"])) & np.isclose(W.budget, 0.10)]
    ang3 = ang3.groupby(["model", "criterion", "bin"]).d.mean().unstack("bin")[BIN_LABELS]
    ang3["worst"] = ang3.min(axis=1)
    out.csv("05_level3", L3)
    out.write("05_level3", "Level 3 - budget-aware fine-tuning",
              ["## PSNR at each budget (100% = no pruning)", md(piv.reset_index(), "{:.2f}"),
               "## All rows with paired deltas (vs full original model, and vs post-hoc hybrid of the original model at the same budget)",
               md(L3),
               "## ΔPSNR vs the original full model per angle bin, 10 % kept", md(ang3.reset_index(), "{:+.2f}")],
              sum([sources.get(p[0], []) for p in plan], []))

    # ---------- 06 cost
    cf = os.path.join(res, "cost", "cost_table.csv")
    if os.path.exists(cf):
        c = pd.read_csv(cf)
        out.write("06_cost", "Cost (storage, render time, render memory)",
                  [md(c.drop(columns=["gpu"]), "{:.3f}"), f"GPU: {c.gpu.iloc[0]}"], [cf])

    # ---------- 08 symmetry x budget
    if "Sym+FT10" in sources or "Sym (post-hoc)" in sources:
        plan = [("original", "hybrid", "plain"), ("Sym (post-hoc)", "hybrid", "symmetric"),
                ("FT10-hybrid", "hybrid", "plain"), ("Sym+FT10", "hybrid", "symmetric"),
                ("AnyBudget", "hybrid", "plain"), ("Sym+AnyBudget", "hybrid", "symmetric")]
        pairs = {"Sym (post-hoc)": "original", "Sym+FT10": "FT10-hybrid", "Sym+AnyBudget": "AnyBudget"}
        rows = []
        full_bytes = 16384 * FLOATS_PER_GAUSSIAN["plain"] * 4
        for m, c, kind in plan:
            for bgt in [0.02, 0.05, 0.10, 0.20, 0.25, 1.0]:
                r = row(m, c, bgt, base=(pairs[m], c, bgt) if m in pairs else None)
                if r:
                    stored = int(round(bgt * 16384))
                    r["stored_gaussians"] = stored
                    r["storage_MB"] = stored * FLOATS_PER_GAUSSIAN[kind] * 4 / 1e6
                    r["compression_vs_original_full"] = full_bytes / (stored * FLOATS_PER_GAUSSIAN[kind] * 4)
                    rows.append(r)
        S = pd.DataFrame(rows)
        piv = S.pivot_table(index="model", columns="budget", values="psnr").reindex([p[0] for p in plan if p[0] in set(S.model)])
        piv.columns = [("100%" if c == 1.0 else f"{c * 100:g}%") for c in piv.columns]
        angS = W[W.model.isin(set(S.model)) & (W.criterion == "hybrid") & np.isclose(W.budget, 0.10)]
        angS = angS.groupby("model").apply(lambda x: x.groupby("bin").d.mean()).reindex(columns=BIN_LABELS)
        angS["worst"] = angS.min(axis=1)
        out.csv("08_symmetry_x_budget", S)
        out.write("08_symmetry_x_budget", "Combination: symmetry (Paul) x budget (Taya)",
                  ["Budget = fraction of STORED (original) Gaussians; in symmetric models each kept original also renders its "
                   "mirrored twin. Storage: 23 floats per Gaussian (plain), 27 (symmetric: + twin switch + colour correction); "
                   "the 4 plane numbers per object are negligible.",
                   "## PSNR at each stored budget", md(piv.reset_index(), "{:.2f}"),
                   "## All rows (paired deltas: vs original full model, and symmetric vs its non-symmetric counterpart)", md(S),
                   "## ΔPSNR vs the original full model per angle bin, 10 % stored", md(angS.reset_index(), "{:+.2f}")],
                  sum([sources.get(p[0], []) for p in plan], []))

    # ---------- 10 ablations of the final model (Sym+AnyBudget) and of the training mask
    abl = [("test_symany_criteria", "test-time rule: other criteria"), ("test_symany_oracle", "TRUE mirror plane"),
           ("test_symany_twinsoff", "twins switched off"), ("test_symany_fp16", "float16 storage"),
           ("test_ft10_random", "FT-10 trained with a RANDOM mask")]
    blocks, srcs, rows = [], [], []
    ref_sym = O[O.model == "Sym+AnyBudget"]
    ref_ft = O[O.model == "FT10-hybrid"]
    for prefix, what in abl:
        v = load_views(res, prefix, what)
        if v is None:
            continue
        srcs += [os.path.basename(f) for f in files_for(res, prefix)]
        v = v[~v.object_idx.isin(blank)]
        o = v.groupby(["criterion", "budget", "object_idx"])[METRICS].mean().reset_index()
        base = ref_ft if prefix == "test_ft10_random" else ref_sym
        base_name = "FT10-hybrid (hybrid)" if prefix == "test_ft10_random" else "Sym+AnyBudget (hybrid, own plane)"
        for (c, bgt), g in o.groupby(["criterion", "budget"]):
            g = g.set_index("object_idx")
            bc = "full" if bgt == 1.0 else "hybrid"
            bb = base[(base.criterion == bc) & np.isclose(base.budget, bgt)].set_index("object_idx")
            common = g.index.intersection(bb.index)
            rows.append({"ablation": what, "criterion": c, "budget": bgt, "n_objects": len(g), "psnr": g.psnr.mean(),
                         "lpips": g.lpips.mean(),
                         "dPSNR_vs_original_full": pm(g.psnr - ref.loc[g.index, "psnr"]),
                         f"dPSNR_vs_reference": pm(g.loc[common].psnr - bb.loc[common].psnr) if len(common) else "—",
                         "reference": base_name})
    if rows:
        T = pd.DataFrame(rows)
        out.csv("10_ablations", T)
        blocks.append(md(T))
    su = os.path.join(res, "sym_usage_summary.csv")
    if os.path.exists(su):
        blocks += ["## How the symmetric models use their twins (fraction of kept originals on the camera side, twin switch)",
                   md(pd.read_csv(su))]
        srcs.append("sym_usage_summary.csv")
    if blocks:
        out.write("10_ablations", "Ablations: final model (any budget + symmetry) and training mask", blocks, srcs)

    # ---------- 09 validation curves (fine-tuning logs)
    tl = sorted(glob.glob(os.path.join(res, "train_logs", "*.out")))
    if tl:
        blocks = []
        for p in tl:
            rows = []
            for line in open(p, errors="ignore"):
                m = re.search(r"\[val\] (start|it (\d+)) \| full ([\d.]+) \| masked ([\d.na]+) \| worst bin ([\d.]+)", line)
                if m:
                    rows.append({"iteration": 0 if m.group(1) == "start" else int(m.group(2)),
                                 "val_full": float(m.group(3)), "val_masked": float(m.group(4)), "val_worst_bin": float(m.group(5))})
                m2 = re.search(r"per budget: (.*)", line)
                if m2 and rows:
                    for kv in m2.group(1).split():
                        k, v = kv.split("=")
                        rows[-1 if "[val] start" not in line else -1][f"val_{k}"] = float(v)
            if rows:
                blocks += [f"## {os.path.basename(p)}", md(pd.DataFrame(rows), "{:.3f}")]
        out.write("09_validation_curves", "Fine-tuning validation curves (20 validation objects; NOT test numbers)", blocks,
                  [os.path.relpath(x, res) for x in tl])

    # ---------- index
    with open(os.path.join(out.folder, "README.md"), "w") as f:
        f.write("# All numbers of the report\n\nGenerated by `code/make_all_tables.py`. Protocol: see `00_protocol.md`.\n\n")
        for name, title in out.index:
            f.write(f"- [{title}]({name}.md)\n")
    print("done:", out.folder)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/final_numbers")
    main(p.parse_args())
