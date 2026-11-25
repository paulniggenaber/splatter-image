# entirely new code file from Taya
"""
Final figures of the report (run on a laptop after make_all_tables.py):

  fig_storage   PSNR and LPIPS vs STORAGE PER OBJECT (MB, log axis) for post-hoc pruning, budget-aware
                fine-tuning, and fine-tuning + symmetry. Storage: 23 floats/Gaussian (plain),
                27 (symmetric: + twin switch + colour correction). This is the fair x-axis for the
                symmetric models, which store fewer bytes than they render.
  fig_budget    Level 1: PSNR vs % kept Gaussians, training-free rules + learned head (no plain contribution)
  fig_angles    ΔPSNR vs the original full model per input-target angle bin at 10 % stored Gaussians.

    python -m src.analysis.make_final_figures --tables results/final_numbers --results results --out results/final_figures
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis import analyze_level1 as A  # noqa: E402

N = 16384
FLOATS = {"plain": 23, "symmetric": 27}
FULL_MB = N * FLOATS["plain"] * 4 / 1e6
BINS = list(range(0, 181, 30))
BIN_LABELS = ["30-60", "60-90", "90-120", "120-150", "150-180"]

# label, table csv, model name in the table, criterion, kind, colour, marker, linewidth
SERIES = [
    ("Original, hybrid",         "03_level1",            "original",      "hybrid",  "plain",     "#e87ba4", "v", 1.4),
    ("Fine-tuned for 10 %",                  "05_level3",            "FT10-hybrid",   "hybrid",  "plain",     "#1baf7a", "D", 1.6),
    ("Fine-tuned for 5 %",                   "05_level3",            "FT5-hybrid",    "hybrid",  "plain",     "#eda100", "^", 1.6),
    ("Fine-tuned for 10 % + symmetry",       "08_symmetry_x_budget", "Sym+FT10",      "hybrid",  "symmetric", "#008300", "X", 1.8),
]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def series_points(tables, tab, model, crit, kind):
    f = os.path.join(tables, tab + ".csv")
    if not os.path.exists(f):
        return None
    t = pd.read_csv(f)
    t = t[(t.model == model) & ((t.criterion == crit) | (t.criterion == "full"))].copy()
    if t.empty:
        return None
    if model == "original":
        t = t[(t.budget <= 0.25) | (t.criterion == "full")]
    t["MB"] = (t.budget * N).round() * FLOATS[kind] * 4 / 1e6
    return t.sort_values("MB")


def fig_storage(a):
    """PSNR vs storage per object (MB, log x): post-hoc pruning vs budget fine-tuning vs + symmetry."""
    full = pd.read_csv(os.path.join(a.tables, "03_level1.csv"))
    full = full[full.criterion == "full"].iloc[0]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.axhline(full["psnr"], color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(0.024, full["psnr"], " original model, all Gaussians (1.51 MB)", color=MUTED, fontsize=7, va="bottom", ha="left")
    for lab, tab, model, crit, kind, col, mk, lw in SERIES:
        t = series_points(a.tables, tab, model, crit, kind)
        if t is None:
            print("missing series:", lab)
            continue
        t = t[t.budget >= 0.02]
        ax.plot(t.MB, t["psnr"], color=col, marker=mk, ms=6, mec="white", mew=0.8, lw=lw, label=lab,
                zorder=3 if "symmetry" in lab else 2)
    ax.set_xscale("log")
    ticks = [0.03, 0.075, 0.15, 0.3, 0.75, 1.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:g}" for x in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.022, 2.1)
    ax.set_ylim(15.5, 24.6)
    ax.set_xlabel("Storage per object (MB)", fontsize=8, color=TEXT)
    ax.set_ylabel("PSNR (dB) ↑", fontsize=8, color=TEXT)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig_storage.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_views(res, prefix, model):
    fs = sorted(glob.glob(os.path.join(res, f"{prefix}*_views.csv")))
    if not fs:
        return None
    v = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["object_idx", "criterion", "budget", "view_id"])
    return v.assign(model=model)


def fig_angles(a):
    runs = [("Original, hybrid", "budget_full_part", "#e87ba4", "v"),
            ("Fine-tuned for 10 %", "test_ft10_hybrid", "#1baf7a", "D"),
            ("Fine-tuned for 10 % + symmetry", "test_sym_ft10", "#008300", "X")]
    orig = load_views(a.results, "budget_full_part", "orig")
    full = orig[orig.criterion == "full"]
    med = full.groupby("object_idx").psnr_masked.median()
    blank = med[med > 30].index
    f0 = full[~full.object_idx.isin(blank)][["object_idx", "view_id", "psnr", "angle_deg"]].rename(columns={"psnr": "p0"})
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.axhline(0, color=MUTED, lw=1)
    for lab, prefix, col, mk in runs:
        v = load_views(a.results, prefix, lab)
        if v is None:
            print("missing:", prefix)
            continue
        v = v[(v.criterion == "hybrid") & np.isclose(v.budget, 0.10)]
        w = v.merge(f0, on=["object_idx", "view_id"], suffixes=("", "_0"))
        w["d"] = w.psnr - w.p0
        w["bin"] = A.bin_angles(w.angle_deg_0.astype(float), BINS)
        per_obj = w.groupby(["bin", "object_idx"]).d.mean().reset_index()
        g = per_obj.groupby("bin").d.agg(["mean", A.ci95]).reindex(BIN_LABELS)
        x = [45, 75, 105, 135, 165]
        ax.errorbar(x, g["mean"], yerr=g["ci95"], color=col, marker=mk, ms=5, mec="white", mew=0.8, lw=1.6,
                    capsize=2, label=lab)
    ax.set_xticks([45, 75, 105, 135, 165])
    ax.set_xlabel("Angle between input and target camera (°)", fontsize=8, color=TEXT)
    ax.set_ylabel("ΔPSNR vs original full model (dB), 10 % stored", fontsize=8, color=TEXT)
    style(ax)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig_angles.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_twins(a):
    """Final model (any budget + symmetry): PSNR with the twins on vs switched off at test time."""
    f8, f10 = os.path.join(a.tables, "08_symmetry_x_budget.csv"), os.path.join(a.tables, "10_ablations.csv")
    if not (os.path.exists(f8) and os.path.exists(f10)):
        print("fig_twins: tables missing"); return
    on = pd.read_csv(f8); on = on[on.model == "Sym+AnyBudget"]
    off = pd.read_csv(f10); off = off[off.ablation == "twins switched off"]
    bud = [0.05, 0.10, 0.20, 1.0]
    get = lambda t, b: t[np.isclose(t.budget, b)].psnr.iloc[0]
    y_on, y_off = [get(on, b) for b in bud], [get(off, b) for b in bud]
    x = np.arange(len(bud))
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(x, y_off, color="#a3a19b", marker="o", ms=6, mec="white", mew=0.8, lw=1.8, label="twins switched off")
    ax.plot(x, y_on, color="#6250d6", marker="P", ms=7, mec="white", mew=0.8, lw=2.2, label="twins on (mirror symmetry)")
    for i in range(len(bud)):
        ax.vlines(x[i], y_off[i], y_on[i], color="#6250d6", lw=0.8, ls=":", zorder=1)
        ax.text(x[i] + 0.08, (y_on[i] + y_off[i]) / 2, f"+{y_on[i] - y_off[i]:.2f} dB", ha="left", va="center", fontsize=7.5, color=TEXT)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(b * 100)} %" for b in bud])
    ax.set_xlabel("Stored Gaussians", fontsize=8, color=TEXT)
    ax.set_ylabel("PSNR (dB) ↑", fontsize=8, color=TEXT)
    ax.set_xlim(-0.3, len(bud) - 0.4)
    style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig_twins.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tables", default="results/final_numbers")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/final_figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_storage(a)
    fig_angles(a)
    fig_twins(a)
    lv1 = pd.read_csv(os.path.join(a.tables, "03_level1.csv"))
    lv1 = lv1[lv1.model == "original"] if "model" in lv1 else lv1
    f2 = os.path.join(a.tables, "04_level2.csv")
    if os.path.exists(f2):   # learned head (Level 2, head_A): 10 / 5 / 1 %
        lv2 = pd.read_csv(f2)
        lv1 = pd.concat([lv1, lv2[(lv2.model == "head_A") & (lv2.criterion == "learned")]], ignore_index=True)
    A.fig_budget(lv1, os.path.join(a.out, "fig_budget"))
    print("figures in", a.out)
