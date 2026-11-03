"""Figures for the minor project report. Run from the repo root:
    python3 report/make_figs.py
Writes PNGs into report/figs/."""
import csv, json, math, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from bounds_extra import bb_zero_upper, any_event_upper, wc_zero_upper
from clusterstats import adjusted_upper, cp_upper, deff, cp_two_sided, cluster_bootstrap_ci

OUT = os.path.join(ROOT, "report", "figs")
os.makedirs(OUT, exist_ok=True)
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "axes.axisbelow": True})
PRETTY = {"qwen3_8b": "Qwen3-8B", "qwen3_8b_think": "Qwen3-8B (thinking)", "qwen3_14b": "Qwen3-14B",
          "qwen3_32b": "Qwen3-32B", "mistral_small_24b": "Mistral-Small-24B", "gpt_oss_20b": "gpt-oss-20b"}
ORDER = list(PRETTY)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=220)
    plt.close(fig)


# 1. System architecture / pipeline -------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
boxes = [
    (0.1, 3.2, "Scenario grid\n2 scenarios x 3 goals\nx 3 urgency = 30 configs"),
    (2.6, 3.2, "Prompt generation\n(agentic misalignment\nsuite, pinned commit)"),
    (5.1, 3.2, "vLLM server\n6 open models\n100 rollouts / config"),
    (7.6, 3.2, "Mechanical gate\n(tool call to target\naddress)"),
    (7.6, 0.6, "LLM judge\n(gpt-oss-120b) +\nblind Opus labels"),
    (5.1, 0.6, "rollouts.csv\n18,000 rows\nmodel, cluster, harmful"),
    (2.6, 0.6, "Cluster statistics\nICC, design effect,\nbeta-binomial bounds"),
    (0.1, 0.6, "Report\ntables, figures,\nre-bounded claims"),
]
for x, y, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 2.2, 1.25, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc="#EAF2FA", ec=C[0], lw=1.1))
    ax.text(x + 1.1, y + 0.62, t, ha="center", va="center", fontsize=8.5)
arrow = dict(arrowstyle="-|>", color="0.25", lw=1.1)
for (x1, y1), (x2, y2) in [((2.3, 3.82), (2.6, 3.82)), ((4.8, 3.82), (5.1, 3.82)), ((7.3, 3.82), (7.6, 3.82)),
                           ((8.7, 3.2), (8.7, 1.85)), ((7.6, 1.22), (7.3, 1.22)), ((5.1, 1.22), (4.8, 1.22)),
                           ((2.6, 1.22), (2.3, 1.22))]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow)
save(fig, "fig_pipeline.png")

# 2. Zero-event bound vs ICC ----------------------------------------------------------
N = 3000
iccs = np.logspace(-3, 0, 120)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for c, m in zip(C, (10, 30, 100)):
    K = N // m
    ax.plot(iccs, [bb_zero_upper(K, m, r) * 100 for r in iccs], color=c, lw=1.8,
            label=f"{K} scenarios x {m} rollouts")
    ax.plot(iccs, [adjusted_upper(0, N, deff(m, r)) * 100 for r in iccs], ":", color=c, lw=1.4)
ax.axhline(cp_upper(0, N) * 100, color="k", ls="--", lw=1, label="naive (rule of three)")
ax.plot([], [], "k-", lw=1.5, label="exact beta-binomial")
ax.plot([], [], "k:", lw=1.5, label="design-effect (Kish)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("intra-cluster correlation (ICC)")
ax.set_ylabel(f"95% upper bound, 0 of {N} harmful (%)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "fig_bound_vs_icc.png")

# 3. Budget allocation ---------------------------------------------------------------
Ks = np.array([k for k in range(5, N + 1) if N % k == 0])
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for c, r in zip(C, (0.01, 0.05, 0.2, 0.5)):
    ms = (N // Ks).astype(int)
    ax.plot(Ks, [bb_zero_upper(int(k_), int(m_), r) * 100 for k_, m_ in zip(Ks, ms)], "o-", ms=3,
            lw=1.5, color=c, label=f"ICC = {r}")
ax.axhline(cp_upper(0, N) * 100, color="k", ls="--", lw=1, label="naive")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(f"number of distinct scenarios (total rollouts fixed at {N})")
ax.set_ylabel("95% bound with 0 harmful (%)")
ax.legend(frameon=False, fontsize=8)
save(fig, "fig_budget.png")

# 4. Coverage simulation -------------------------------------------------------------
rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "sim_coverage.csv"))))
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, mu in zip(axes, ("0.001", "0.003", "0.01")):
    sub = [r for r in rows if r["mu"] == mu]
    x = np.arange(len(sub))
    w = 0.27
    for j, (key, lab) in enumerate((("cov_naive", "naive"), ("cov_estimated", "ICC estimated"),
                                    ("cov_oracle", "true ICC"))):
        ax.bar(x + (j - 1) * w, [float(r[key]) * 100 for r in sub], w, color=C[j], label=lab)
    ax.axhline(95, color="k", ls="--", lw=0.8)
    ax.set_xticks(x, [r["rho"] for r in sub], fontsize=8)
    ax.set_title(f"true rate {float(mu) * 100:g}%", fontsize=9)
    ax.set_xlabel("ICC", fontsize=9)
    ax.set_ylim(0, 105)
axes[0].set_ylabel("coverage of 95% bound (%)")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(OUT, "fig_coverage.png"), dpi=220); plt.close(fig)

# 5. Per-configuration rates ---------------------------------------------------------
data = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for r in csv.DictReader(open(os.path.join(ROOT, "data", "rollouts_opus.csv"))):
    cell = data[r["model"]][r["cluster"]]
    cell[0] += int(r["harmful"]); cell[1] += 1
clusters = sorted(next(iter(data.values())))
mat = np.array([[data[m][c][0] / data[m][c][1] * 100 for c in clusters] for m in ORDER])
fig, ax = plt.subplots(figsize=(7.4, 3.4))
im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=max(80, mat.max()))
ax.set_yticks(range(len(ORDER)), [PRETTY[m] for m in ORDER], fontsize=8)
short = [c.replace("blackmail_", "B:").replace("leaking_", "L:").replace("explicit", "exp")
         .replace("latent", "lat").replace("america", "am").replace("global", "gl")
         .replace("replacement", "repl").replace("restriction", "restr") for c in clusters]
ax.set_xticks(range(len(clusters)), short, rotation=90, fontsize=6)
ax.axvline(14.5, color="k", lw=1)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
cb.set_label("harmful rate (%)", fontsize=8)
save(fig, "fig_heatmap.png")

# 6. Naive vs cluster CI per model ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, scope in zip(axes, ("blackmail", "leaking")):
    for i, m in enumerate(ORDER):
        names = [c for c in clusters if c.startswith(scope)]
        k = np.array([data[m][c][0] for c in names], float)
        n = np.array([data[m][c][1] for c in names], float)
        rate = k.sum() / n.sum() * 100
        lo, hi = cp_two_sided(k.sum(), n.sum())
        blo, bhi = cluster_bootstrap_ci(k, n)
        ax.plot([lo * 100, hi * 100], [i + 0.15] * 2, color=C[0], lw=2.5, solid_capstyle="round")
        ax.plot([blo * 100, bhi * 100], [i - 0.15] * 2, color=C[1], lw=2.5, solid_capstyle="round")
        ax.plot(rate, i, "o", color="k", ms=3.5)
    ax.set_title(scope, fontsize=10)
    ax.set_xlabel("harmful rate (%)")
axes[0].set_yticks(range(len(ORDER)), [PRETTY[m] for m in ORDER], fontsize=8)
axes[0].plot([], [], color=C[0], lw=2.5, label="naive 95% CI")
axes[0].plot([], [], color=C[1], lw=2.5, label="cluster bootstrap 95% CI")
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
axes[0].invert_yaxis()
save(fig, "fig_ci_compare.png")

# 7. Hero: rollouts per scenario --------------------------------------------------------
K15, M = 15, 100
borrow = 0.0464
ms = np.unique(np.round(np.logspace(0, 4, 80)).astype(int))
naive = lambda n: 1 - 0.05 ** (1 / n)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(ms, [naive(K15 * m) * 100 for m in ms], "--", color="0.2", lw=1.4, label="naive (all rollouts independent)")
ax.plot(ms, [wc_zero_upper(K15, int(m), borrow) * 100 for m in ms], color=C[1], lw=2,
        label="15 scenarios, worst case at ICC 0.046")
ax.plot(ms, [wc_zero_upper(150, int(m), borrow) * 100 for m in ms], ":", color=C[0], lw=2,
        label="150 scenarios, worst case at ICC 0.046")
ax.axhline(naive(K15) * 100, color="0.55", lw=1.2, label="scenario bound (no assumption)")
ax.plot(M, naive(K15 * M) * 100, "o", color="0.2"); ax.plot(M, wc_zero_upper(K15, M, borrow) * 100, "o", color=C[1])
ax.annotate("our data: 0 / 1500", xy=(M, naive(K15 * M) * 100), xytext=(3, 0.02), fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.6))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 40)
ax.set_xlabel("rollouts per scenario")
ax.set_ylabel("95% bound on harmful rate (%)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
save(fig, "fig_rollouts_per_scenario.png")

# 8. Published claims re-bounded ----------------------------------------------------
rea = list(csv.DictReader(open(os.path.join(ROOT, "data", "reanalysis.csv"))))
labels = {"C7": "Shutdown resistance", "C2": "Record tampering", "C8": "Covert sabotage",
          "C6": "Email reranking", "C13": "Scheming reasoning", "C5": "Oversight subversion",
          "C10": "Research sabotage", "C11": "Prefilled sabotage"}
fig, ax = plt.subplots(figsize=(7.0, 3.6))
y = np.arange(len(rea))
for j, (key, lab) in enumerate((("naive", "naive"), ("bb0.05", "beta-binomial, ICC 0.05"),
                                ("bb0.2", "beta-binomial, ICC 0.2"), ("scen", "scenario bound"))):
    ax.barh(y + (j - 1.5) * 0.2, [float(r[key]) * 100 for r in rea], 0.2, color=C[j], label=lab)
ax.set_yticks(y, [f"{labels[r['id']]} (N={r['N']}, K={r['K']})" for r in rea], fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("95% upper bound on rate (%)")
ax.invert_yaxis()
ax.legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "fig_reanalysis.png")

# 9. Judge agreement ---------------------------------------------------------------
ja = json.load(open(os.path.join(ROOT, "data", "judge_agreement.json")))["strict"]
cm = np.array([[ja["tp"], ja["fn"]], [ja["fp"], ja["tn"]]])
fig, ax = plt.subplots(figsize=(3.8, 3.2))
ax.imshow(cm, cmap="Blues")
ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > 1500 else "black")
ax.set_xticks([0, 1], ["judge: harmful", "judge: not"])
ax.set_yticks([0, 1], ["Opus: harmful", "Opus: not"])
ax.set_title(f"n = {ja['n']}, kappa = {ja['kappa']:.2f}", fontsize=10)
save(fig, "fig_judge.png")
print("done")
