# /// script
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Figure of per-configuration rates and the re-bounded zero-event claims from our runs.

Reads data/rollouts_opus.csv (Opus labels; pass another CSV as argv[1]). Writes paper/figs/real_rates.pdf, paper/tab_zero.tex and
paper/numbers_zero.tex.
Run: uv run analysis/real_figs.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_real import PRETTY, tex_name
from bounds_extra import any_event_upper, bb_upper
from clusterstats import adjusted_upper, cp_upper, deff, icc_anova

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = ["#0072B2", "#D55E00"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

cells = defaultdict(lambda: [0, 0])
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "rollouts_opus.csv")
for r in csv.DictReader(open(SRC)):
    c = cells[(r["model"], r["cluster"])]
    c[0] += int(r["harmful"]); c[1] += 1
models = [m for m in PRETTY if any(k[0] == m for k in cells)]

# ---- Figure: per-configuration harmful rate --------------------------------------
# One row per model and behaviour. Configurations with zero events are drawn as one
# hollow marker at 0 with their count beside it, so they do not hide each other.
# Configurations with the same count are stacked vertically. The grey bar is the
# central 95% range of a configuration's rate (Binomial(100, p) / 100) if every
# configuration of that model and behaviour shared the pooled rate p: points outside
# it vary more than sampling noise alone would make them.
from scipy import stats

fig, ax = plt.subplots(figsize=(6.8, 3.4))
outside = total = 0
for i, m in enumerate(models):
    for j, scen in enumerate(("blackmail", "leaking")):
        kn = [(k, n) for (mm, c), (k, n) in cells.items() if mm == m and c.startswith(scen)]
        y0 = i + (j - 0.5) * 0.36
        p = sum(k for k, _ in kn) / sum(n for _, n in kn)
        n0 = int(np.mean([n for _, n in kn]))
        lo, hi = stats.binom.ppf([0.025, 0.975], n0, p) / n0 * 100
        ax.plot([lo, hi], [y0, y0], color="0.6", lw=5, alpha=0.35, solid_capstyle="butt", zorder=1)
        outside += sum(not (lo <= k / n * 100 <= hi) for k, n in kn)
        total += len(kn)
        zeros = sum(k == 0 for k, _ in kn)
        if zeros:
            ax.scatter([0], [y0], s=16, facecolor="white", edgecolor=C[j], lw=1.0, zorder=3)
            ax.text(-1.2, y0, f"{zeros}", ha="right", va="center", fontsize=6.5, color="0.25")
        seen = {}
        for k, n in sorted(kn):
            if k == 0:
                continue
            r = k / n * 100
            off = seen.get(r, 0)
            seen[r] = off + 1
            dy = 0.07 * ((off + 1) // 2) * (1 if off % 2 else -1)  # 0, +1, -1, +2, ... steps
            ax.scatter([r], [y0 + dy], s=12, color=C[j], edgecolor="white", lw=0.4, zorder=3)
ax.set_yticks(range(len(models)), [PRETTY[m] for m in models])
ax.invert_yaxis()
ax.set_xlim(-5, None)
ax.set_xlabel("harmful rollouts in a configuration (%), 100 rollouts each")
ax.grid(axis="x", color="0.92", lw=0.5)
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", color=C[0], ms=4, label="blackmail"),
           Line2D([], [], marker="o", ls="", color=C[1], ms=4, label="leaking"),
           Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", ms=4, label="no harmful rollout (count beside it)"),
           Line2D([], [], color="0.6", lw=5, alpha=0.35, label="95% range if all shared one rate")]
ax.legend(handles=handles, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "paper", "figs", "real_rates.pdf"))
print(f"configurations outside the shared-rate 95% range: {outside} of {total}")

# ---- Zero-event claims from our own runs ----------------------------------------
# rho is not identified with zero events, so we borrow it. The only blackmail cell
# with enough events for an estimate is Mistral-Small-24B; we use that estimate,
# the upper end of its interval, and twice the point estimate.
k_m = np.array([cells[("mistral_small_24b", c)][0] for (mm, c) in cells if mm == "mistral_small_24b" and c.startswith("blackmail")], float)
n_m = np.array([cells[("mistral_small_24b", c)][1] for (mm, c) in cells if mm == "mistral_small_24b" and c.startswith("blackmail")], float)
rho_b = icc_anova(k_m, n_m)
macros = {"BorrowRho": f"{rho_b:.3f}", "BorrowRhoDouble": f"{2 * rho_b:.3f}"}
rows = []
for m in models:
    ks = [(k, n) for (mm, c), (k, n) in cells.items() if mm == m and c.startswith("blackmail")]
    x = sum(k for k, _ in ks)
    if x:
        continue
    sizes = [n for _, n in ks]
    N, K = sum(sizes), len(sizes)
    row = dict(model=PRETTY[m], N=N, K=K, naive=cp_upper(0, N), scen=any_event_upper(0, K),
               bb=bb_upper(0, sizes, rho_b), bb2=bb_upper(0, sizes, 2 * rho_b),
               kish=adjusted_upper(0, N, deff(np.mean(sizes), rho_b)))
    rows.append(row)
    print(row)

# All zero-event cells share N = 1500 and K = 15, so the bounds are the same for each.
assert len({(r["N"], r["K"]) for r in rows}) == 1
r0 = rows[0]
macros["ZeroModels"] = ", ".join(r["model"] for r in rows)
macros["ZeroCount"] = str(len(rows))
for key, name in (("naive", "Naive"), ("scen", "Scen"), ("bb", "Exact"), ("bb2", "ExactDouble"), ("kish", "Kish")):
    macros[f"ZeroUB{name}"] = f"{r0[key] * 100:.2f}"
with open(os.path.join(ROOT, "paper", "tab_zero.tex"), "w") as fh:
    fh.write("% generated by analysis/real_figs.py; do not edit\n")
    fh.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n"
             "$N$ & $K$ & naive & Kish, $\\rho{=}" + f"{rho_b:.3f}" + "$ & exact, $\\rho{=}" + f"{rho_b:.3f}" +
             "$ & exact, $\\rho{=}" + f"{2 * rho_b:.3f}" + "$ & scenarios \\\\\n\\midrule\n")
    fh.write(f"{r0['N']} & {r0['K']} & {r0['naive'] * 100:.2f} & {r0['kish'] * 100:.2f} & "
             f"{r0['bb'] * 100:.2f} & {r0['bb2'] * 100:.2f} & {r0['scen'] * 100:.1f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(ROOT, "paper", "numbers_zero.tex"), "w") as fh:
    fh.write("% generated by analysis/real_figs.py; do not edit\n")
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(macros)
