"""
Analyse the Level 1 CSVs written by eval_budget.py (all parts are merged automatically).

    python analyze_level1.py results/budget_full_part --out results/analysis
    (prefix = everything before "0_objects.csv"; also works for a single run: results/smoke)

Writes to --out:
    summary.csv / summary.md       mean metric + paired delta vs full model (mean +- 95% CI) per criterion x budget
    table_psnr.tex                 LaTeX table (booktabs) for the paper: PSNR / masked PSNR / LPIPS at key budgets
    angle_delta.csv / .md          paired delta-PSNR vs full model per 30-degree angle bin at chosen budgets
    fig_budget.pdf/.png            quality vs fraction of kept Gaussians (PSNR and LPIPS)
    fig_angle.pdf/.png             delta-PSNR vs input-target angle at --angle_budget
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference categorical palette (fixed order, never cycled); random = neutral gray control.
STYLE = {
    "contribution":          dict(color="#2a78d6", label="Contribution (oracle)", marker="o"),
    "opacity":               dict(color="#eb6834", label="Opacity", marker="s"),
    "opacity_area":          dict(color="#1baf7a", label="Opacity × area", marker="D"),
    "contribution_iter_bal": dict(color="#eda100", label="Contribution, iterative + balanced (oracle)", marker="^"),
    "hybrid":                dict(color="#e87ba4", label="Hybrid rank (opacity + opacity × area)", marker="v"),
    "learned":               dict(color="#6250d6", label="Learned head (ours)", marker="P"),
    "merge":                 dict(color="#008300", label="Merge b×b (ToMe-style)", marker="X"),
    "random":                dict(color="#8a8984", label="Random", marker="."),
}
EXTRA_LABELS = {"scale": "Scale (volume)",
                "error_sens": "Error sensitivity (oracle)",
                "edge_opacity": "Edge + opacity (PhGS-style)", "error_sens_signed": "Error sensitivity, signed (oracle)",
                "contribution_bal": "Contribution, angle-balanced",
                "contribution_iter": "Contribution, iterative",
                "contribution_iter_bal": "Contribution, iterative + balanced (oracle)"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["full", "random", "opacity", "scale", "opacity_area", "hybrid", "edge_opacity",
         "contribution", "contribution_bal", "contribution_iter", "contribution_iter_bal",
         "error_sens", "error_sens_signed", "merge", "learned"]
METRICS = ["psnr", "psnr_masked", "ssim", "lpips"]


def load(prefix, kind):
    files = sorted(glob.glob(f"{prefix}*_{kind}.csv"))
    if not files:
        raise SystemExit(f"no files match {prefix}*_{kind}.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = ["object_idx", "criterion", "budget"] + (["view_id"] if kind == "views" else [])
    return df.drop_duplicates(key), files


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def summarise(obj):
    full = obj[obj.criterion == "full"].set_index("object_idx")[METRICS]
    rows = []
    for (crit, budget), g in obj.groupby(["criterion", "budget"]):
        g = g.set_index("object_idx")
        r = dict(criterion=crit, budget=budget, n_objects=len(g),
                 n_gaussians=int(g.n_gaussians.median()))
        for m in METRICS:
            d = g[m] - full.loc[g.index, m]
            r[m] = g[m].mean()
            r[f"d_{m}"] = d.mean()
            r[f"d_{m}_ci"] = ci95(d)
        r["select_ms"] = g.select_ms.iloc[1:].mean() if len(g) > 1 else g.select_ms.mean()  # skip warm-up obj
        rows.append(r)
    s = pd.DataFrame(rows)
    s["order"] = s.criterion.map({c: i for i, c in enumerate(ORDER)}).fillna(99)
    return s.sort_values(["order", "budget"], ascending=[True, False]).drop(columns="order")


def to_md(df, cols, floatfmt=3):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    lines = []
    for _, r in df[cols].iterrows():
        cells = [f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def latex_table(s, budgets, path):
    crits = [c for c in ORDER if c in set(s.criterion) and c not in ("full", "merge")]
    full = s[s.criterion == "full"].iloc[0]
    cols = "l" + "c" * len(budgets)
    out = ["\\begin{table*}[t]", "\\centering", "\\small",
           f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
           "Criterion & " + " & ".join(f"{int(round(b * 100))}\\%" for b in budgets) + " \\\\",
           "\\midrule",
           f"Full model (100\\%) & \\multicolumn{{{len(budgets)}}}{{c}}{{{full.psnr:.2f}}} \\\\",
           "\\midrule"]
    best = {b: s[(s.budget == b) & s.criterion.isin(crits)].psnr.max() for b in budgets}
    for c in crits:
        cells = []
        for b in budgets:
            r = s[(s.criterion == c) & np.isclose(s.budget, b)]
            if r.empty:
                cells.append("--"); continue
            v = r.psnr.iloc[0]
            txt = f"{v:.2f}"
            cells.append(f"\\textbf{{{txt}}}" if txt == f"{best[b]:.2f}" else txt)
        name = STYLE.get(c, {}).get("label", EXTRA_LABELS.get(c, c.replace("_", " ")))
        name = name.replace("×", "$\\times$")
        out.append(f"{name} & " + " & ".join(cells) + " \\\\")
    m = s[s.criterion == "merge"]
    if not m.empty:
        out.append("\\midrule")
        for _, r in m.sort_values("budget", ascending=False).iterrows():
            out.append(f"Merge (keeps {r.budget * 100:.2f}\\%) & \\multicolumn{{{len(budgets)}}}{{c}}{{{r.psnr:.2f}}} \\\\")
    out += ["\\bottomrule", "\\end{tabular}",
            "\\caption{Novel-view PSNR on SRN Cars (704 test objects) when keeping a fraction of the "
            "16{,}384 predicted Gaussians, for each selection criterion. Best per budget in bold.}",
            "\\label{tab:level1}", "\\end{table*}"]
    open(path, "w").write("\n".join(out) + "\n")


def style_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def fig_budget(s, path):
    full = s[s.criterion == "full"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, m, ylab in [(axes[0], "psnr", "PSNR (dB) ↑"), (axes[1], "lpips", "LPIPS ↓")]:
        ax.axhline(full[m], color=MUTED, linewidth=1, linestyle="--", zorder=1)
        ax.text(1, full[m], " full model", color=MUTED, fontsize=7, va="bottom" if m == "psnr" else "top", ha="left")
        for c, st in STYLE.items():
            g = s[s.criterion == c].sort_values("budget")
            if g.empty:
                continue
            kw = dict(color=st["color"], marker=st["marker"], markersize=5,
                      markeredgecolor="white", markeredgewidth=0.8, label=st["label"], zorder=2 if c == "random" else 3)
            if c == "merge":
                ax.plot(g.budget * 100, g[m], linestyle="none", markersize=8, **{k: v for k, v in kw.items() if k != "markersize"})
            else:
                ax.plot(g.budget * 100, g[m], linewidth=1.6, **kw)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 25]); ax.set_xticklabels(["1", "2", "5", "10", "25"])
        ax.set_xlabel("Gaussians kept (%)", fontsize=8, color=TEXT)
        ax.set_ylabel(ylab, fontsize=8, color=TEXT)
        style_axes(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def bin_angles(a, bins):
    lab = [f"{lo}-{hi}" for lo, hi in zip(bins[:-1], bins[1:])]
    return pd.cut(a, bins=bins, include_lowest=True, labels=lab).astype(str)


def angle_table(views, budgets, bins):
    full = views[views.criterion == "full"][["object_idx", "view_id", "psnr", "angle_deg"]]
    v = views[views.criterion != "full"].merge(full, on=["object_idx", "view_id"], suffixes=("", "_full"))
    v["d_psnr"] = v.psnr - v.psnr_full
    v["angle_bin"] = bin_angles(v.angle_deg_full, bins)
    v = v[np.isclose(v.budget.values[:, None], np.array(budgets)[None]).any(1) | (v.criterion == "merge")]
    # per-object mean inside each bin first, then CI across objects (views of one object are correlated)
    per_obj = v.groupby(["criterion", "budget", "angle_bin", "object_idx"]).d_psnr.mean().reset_index()
    t = per_obj.groupby(["criterion", "budget", "angle_bin"]).d_psnr.agg(
        ["mean", ci95, "count"]).reset_index().rename(columns={"ci95": "ci", "count": "n_objects"})
    fullbin = full.assign(angle_bin=bin_angles(full.angle_deg, bins)) \
        .groupby("angle_bin").psnr.mean().rename("psnr_full").reset_index()
    t = t.merge(fullbin, on="angle_bin")
    t["angle_mid"] = t.angle_bin.map(lambda b: (float(b.split("-")[0]) + float(b.split("-")[1])) / 2)
    return t.sort_values(["criterion", "budget", "angle_mid"])


def fig_angle(t, budget, path, crits=("opacity", "opacity_area", "hybrid", "contribution", "learned")):
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    for c in crits:
        g = t[(t.criterion == c) & np.isclose(t.budget, budget)]
        if g.empty:
            continue
        st = STYLE[c]
        x = g.angle_mid
        ax.errorbar(x, g["mean"], yerr=g["ci"], color=st["color"], marker=st["marker"], markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8, linewidth=1.6, capsize=2,
                    label=st["label"], zorder=3)
    ax.set_xlabel("Angle between input and target camera (°)", fontsize=8, color=TEXT)
    ax.set_ylabel(f"ΔPSNR vs full model (dB), {budget * 100:g}% kept", fontsize=8, color=TEXT)
    ax.set_xticks([45, 75, 105, 135, 165])
    style_axes(ax)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def drop_blank_objects(views, thresh=30.0):
    """Drop test objects whose ground truth contains (almost) no object pixels. In SRN Cars this is
    car 375 (876d92ce6a0e4bf399588eee976baae): all 251 images are blank, so every method scores
    ~40-75 dB on it. Detected as median masked PSNR of the full model > thresh (normal cars: <= 25)."""
    full = views[views.criterion == "full"]
    med = full.groupby("object_idx").psnr_masked.median()
    blank = med[med > thresh].index.tolist()
    print(f"blank test objects dropped entirely: {blank}")
    return views[~views.object_idx.isin(blank)]


def drop_empty_views(views):
    """Views whose ground truth is pure background (no object pixel -> masked PSNR = 100) make PSNR
    infinite as soon as a pruned render is also empty. Drop them for ALL methods (paired comparison)."""
    views = drop_blank_objects(views)
    full = views[views.criterion == "full"]
    empty = full[full.psnr_masked >= 99.9][["object_idx", "view_id"]]
    v = views.merge(empty.assign(_empty=1), on=["object_idx", "view_id"], how="left")
    print(f"empty-GT views dropped: {len(empty)} of {len(full)} "
          f"(objects: {sorted(empty.object_idx.unique().tolist())})")
    return v[v._empty.isna()].drop(columns="_empty")


def objects_from_views(views, obj):
    """Recompute per-object means from the (cleaned) per-view rows; keep counts/timing from obj."""
    m = views.groupby(["object_idx", "criterion", "budget"])[METRICS].mean().reset_index()
    keep = obj[["object_idx", "criterion", "budget", "n_gaussians", "select_ms"]]
    return m.merge(keep, on=["object_idx", "criterion", "budget"])


def main(a):
    os.makedirs(a.out, exist_ok=True)
    obj, files = load(a.prefix, "objects")
    print(f"objects files: {len(files)}  | objects: {obj.object_idx.nunique()}  | rows: {len(obj)}")
    views = None
    if not a.skip_views:
        views, vfiles = load(a.prefix, "views")
        print(f"views files: {len(vfiles)} | rows: {len(views)}")
        views = drop_empty_views(views)
        obj = objects_from_views(views, obj)
    s = summarise(obj)
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    cols = ["criterion", "budget", "n_gaussians", "psnr", "d_psnr", "d_psnr_ci", "psnr_masked",
            "d_psnr_masked", "ssim", "lpips", "d_lpips", "select_ms"]
    md = to_md(s, cols)
    open(os.path.join(a.out, "summary.md"), "w").write(md)
    print(md)
    latex_table(s, a.table_budgets, os.path.join(a.out, "table_psnr.tex"))
    fig_budget(s, os.path.join(a.out, "fig_budget"))

    if views is not None:
        t = angle_table(views, a.angle_budgets, list(range(0, 181, 30)))
        t.to_csv(os.path.join(a.out, "angle_delta.csv"), index=False)
        tm = t[t.criterion.isin(["opacity", "opacity_area", "contribution", "error_sens", "learned", "merge"])].copy()
        amd = to_md(tm, ["criterion", "budget", "angle_bin", "n_objects", "psnr_full", "mean", "ci"])
        open(os.path.join(a.out, "angle_delta.md"), "w").write(amd)
        print(amd)
        fig_angle(t, a.angle_budget, os.path.join(a.out, "fig_angle"))
    print("written to", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("prefix", help='e.g. results/budget_full_part (files <prefix>*_objects.csv)')
    p.add_argument("--out", default="results/analysis")
    p.add_argument("--table_budgets", nargs="+", type=float, default=[0.25, 0.10, 0.05, 0.02, 0.01])
    p.add_argument("--angle_budgets", nargs="+", type=float, default=[0.10, 0.05])
    p.add_argument("--angle_budget", type=float, default=0.10, help="budget shown in fig_angle")
    p.add_argument("--skip_views", action="store_true")
    main(p.parse_args())
