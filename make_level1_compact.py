# entirely new code file from Taya
"""Compact Level-1 table for the paper: PSNR (dB) / LPIPS of training-free selection rules at 10/5/2 %
of the Gaussians, original model, 703 cars, 25 views.
    python code/make_level1_compact.py --table results/final_numbers/03_level1.csv --out results/final_numbers/table_level1_compact.tex
"""
import argparse
import numpy as np
import pandas as pd

ROWS = [("random", "Random"), ("opacity", "Opacity"), ("scale", "Scale"), ("edge_opacity", "Edge + opacity [PhGS]"),
        ("opacity_area", "Opacity $\\times$ area (ours)"), ("hybrid", "Hybrid (ours)"),
        ("contribution_iter_bal", "Contribution, iter.\\ + bal.$^\\dagger$ [LightGaussian]"), ("merge", "Merge 4$\\times$4 [ToMe]$^\\ddagger$")]
BUDGETS = [0.10, 0.05, 0.02]


def cell(t, c, b):
    bb = 0.0625 if c == "merge" and b == 0.05 else b
    r = t[(t.criterion == c) & np.isclose(t.budget, bb)]
    return None if r.empty else r.iloc[0]


def main(a):
    t = pd.read_csv(a.table)
    full = t[t.criterion == "full"].iloc[0]
    best = {b: max(r.psnr for c, _ in ROWS if (r := cell(t, c, b)) is not None and c not in ("contribution_iter_bal", "merge")) for b in BUDGETS}
    lines = ["\\begin{table}[t]\\centering\\small", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{l" + "cc" * len(BUDGETS) + "}", "\\toprule",
             "Kept Gaussians & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{int(b*100)}\\%}}" for b in BUDGETS) + " \\\\",
             " & " + " & ".join(["PSNR$\\uparrow$ & LPIPS$\\downarrow$"] * len(BUDGETS)) + " \\\\", "\\midrule"]
    for c, name in ROWS:
        cells = []
        for b in BUDGETS:
            r = cell(t, c, b)
            if r is None:
                cells += ["--", "--"]
            else:
                p = f"{r.psnr:.2f}"
                if c not in ("contribution_iter_bal", "merge") and np.isclose(r.psnr, best[b]):
                    p = f"\\textbf{{{p}}}"
                cells += [p, f"{r.lpips:.3f}"]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\midrule", f"All 16,384 Gaussians & \\multicolumn{{{2*len(BUDGETS)}}}{{c}}{{{full.psnr:.2f} / {full.lpips:.3f}}} \\\\",
              "\\bottomrule", "\\end{tabular}",
              "\\caption{Training-free selection on the original model (703 cars, 25 novel views each). "
              "$^\\dagger$Oracle: needs renders of other views. $^\\ddagger$Merging reaches only 25\\,\\% and "
              "6.25\\,\\% (6.25\\,\\% shown in the 5\\,\\% column, not bolded: it keeps more Gaussians).}", "\\label{tab:level1}", "\\end{table}"]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="results/final_numbers/03_level1.csv")
    p.add_argument("--out", default="results/final_numbers/table_level1_compact.tex")
    main(p.parse_args())
