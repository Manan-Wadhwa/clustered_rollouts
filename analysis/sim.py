# /// script
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Simulation and analytic figures. Needs no real rollouts, so the paper still
has results if the GPU runs fail.

Writes paper/figs/bound_vs_icc.pdf, paper/figs/budget.pdf, paper/numbers_sim.tex,
paper/tab_coverage.tex and data/sim_coverage.csv. (paper/figs/coverage.pdf now comes
from sim_robust.py.)
Run: uv run analysis/sim.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

from bounds_extra import bb_zero_upper, log_p_zero
from clusterstats import adjusted_upper, cp_upper, deff, icc_anova, icc_betabinom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
# Okabe-Ito, colour-blind safe
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

macros = {}


def macro(name, value):
    macros[name] = value


# ---- Figure 1: zero-event upper bound vs ICC at fixed N -----------------------
# Solid: exact beta-binomial bound (the one the text explains through m_eff);
# dotted: the Kish bound for comparison.
N = 3000
iccs = np.logspace(-3, 0, 200)
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for c, m in zip(C, (10, 30, 100)):
    K = N // m
    ax.plot(iccs, [bb_zero_upper(K, m, r) * 100 for r in iccs], color=c, lw=1,
            label=f"{K} scenarios x {m} rollouts")
    ax.plot(iccs, [adjusted_upper(0, N, deff(m, r)) * 100 for r in iccs], ":", color=c, lw=1)
naive = cp_upper(0, N)
ax.axhline(naive * 100, color="k", ls="--", lw=0.8, label="naive (rule of three)")
ax.plot([], [], "k-", lw=1, label="beta-binomial")
ax.plot([], [], "k:", lw=1, label="design effect")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("intra-cluster correlation (ICC)")
ax.set_ylabel(f"95% bound with 0 of {N} (%)")
ax.legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "bound_vs_icc.pdf"))

macro("SimN", f"{N:,}".replace(",", "{,}"))
macro("NaiveUB", f"{naive * 100:.2f}")
for m, mname in ((30, "Thirty"), (100, "Hundred")):
    for r, tag in ((0.01, "A"), (0.1, "B")):
        d = deff(m, r)
        macro(f"Deff{mname}{tag}", f"{d:.2f}")
        macro(f"UB{mname}{tag}", f"{adjusted_upper(0, N, d) * 100:.2f}")
macro("UBthirtyClusters", f"{cp_upper(0, 30) * 100:.1f}")

# ---- Figure 2: budget allocation at fixed N -----------------------------------
Ks = np.array([k for k in range(5, N + 1) if N % k == 0])
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for c, r in zip(C, (0.01, 0.05, 0.2, 0.5)):
    ms = (N // Ks).astype(int)
    ax.plot(Ks, [bb_zero_upper(int(k_), int(m_), r) * 100 for k_, m_ in zip(Ks, ms)], "o-", ms=2,
            lw=1, color=c, label=f"ICC = {r}")
    ax.plot(Ks, [adjusted_upper(0, N, deff(m_, r)) * 100 for m_ in ms], ":", lw=1, color=c)
ax.axhline(naive * 100, color="k", ls="--", lw=0.8, label="naive")
ax.plot([], [], "k-", lw=1, label="beta-binomial")
ax.plot([], [], "k:", lw=1, label="design effect")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(f"number of scenarios (total rollouts fixed at {N})")
ax.set_ylabel("95% bound with 0 events (%)")
ax.set_ylim(top=ax.get_ylim()[1] * 6)  # room for the legend above the curves
ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "budget.pdf"))

# ---- Coverage simulation ------------------------------------------------------
# True scenario-level rates p_i ~ Beta(mean mu, ICC rho); k_i ~ Binomial(m, p_i).
# Target: the scenario-average rate mu. Coverage = P(mu <= one-sided 95% upper bound).
# REPS was 4,000 in the first draft; its zero-event share at rate 0.1%, ICC 0.01 came out
# 2.4 Monte Carlo SEs above the exact value (seed noise, same model), so we use 20,000.
rng = np.random.default_rng(20261022)
REPS = 20000
K, m = 30, 100


def cp_upper_vec(x, n, alpha=0.05):
    """clusterstats.cp_upper, elementwise over arrays x and n."""
    x, n = np.broadcast_arrays(np.asarray(x, float), np.asarray(n, float))
    out = np.ones_like(x)
    ok = x < n
    out[ok] = stats.beta.ppf(1 - alpha, x[ok] + 1, n[ok] - x[ok])
    return out


def icc_anova_rows(k, m):
    """clusterstats.icc_anova for each row of k (equal cluster sizes m), nan with
    no events or no non-events."""
    K_ = k.shape[1]
    n = K_ * m
    p = k.sum(1) / n
    pi = k / m
    msb = m * ((pi - p[:, None]) ** 2).sum(1) / (K_ - 1)
    msw = m * (pi * (1 - pi)).sum(1) / (n - K_)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (msb - msw) / (msb + (m - 1) * msw)  # m0 = m for equal sizes
    return np.where((p > 0) & (p < 1), r, np.nan)


rows = []
for mu in (0.001, 0.003, 0.01):
    for rho in (0.0, 0.01, 0.05, 0.2, 0.5):
        if rho == 0:
            p = np.full((REPS, K), mu)
        else:
            a, b = mu * (1 - rho) / rho, (1 - mu) * (1 - rho) / rho
            p = rng.beta(a, b, (REPS, K))
        k = rng.binomial(m, p)
        x, n = k.sum(1), K * m
        # ICC estimated from the same data; falls back to naive when it is
        # not estimable (no events), which is exactly the zero-event case.
        r_hat = icc_anova_rows(k, m)
        for i in range(0, REPS, 997):  # spot-check against the scalar estimator
            ref = icc_anova(k[i], np.full(K, m))
            assert (np.isnan(ref) and np.isnan(r_hat[i])) or abs(ref - r_hat[i]) < 1e-10
        d_hat = np.where(np.isnan(r_hat), 1.0, deff(m, np.maximum(np.nan_to_num(r_hat), 0.0)))
        d = deff(m, rho)
        pz_exact = float(np.exp(log_p_zero(mu, np.full(K, m), rho)))
        se = np.sqrt(max(pz_exact * (1 - pz_exact), 1e-6) / REPS)
        rows.append(dict(mu=mu, rho=rho, K=K, m=m, reps=REPS,
                         p_zero=float((x == 0).mean()), p_zero_exact=pz_exact,
                         z_p_zero=float(((x == 0).mean() - pz_exact) / se),
                         cov_naive=float((cp_upper_vec(x, n) >= mu).mean()),
                         cov_oracle=float((cp_upper_vec(x / d, n / d) >= mu).mean()),
                         cov_estimated=float((cp_upper_vec(x / d_hat, n / d_hat) >= mu).mean())))
        print(rows[-1])

with open(os.path.join(ROOT, "data", "sim_coverage.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)


with open(os.path.join(ROOT, "paper", "tab_coverage.tex"), "w") as fh:
    fh.write("% generated by analysis/sim.py; do not edit\n")
    fh.write("\\begin{tabular}{rrrrrr}\n\\toprule\n"
             "true rate (\\%) & ICC & evaluations with 0 events (\\%) & naive & $\\hat\\rho$ from data & true $\\rho$ \\\\\n"
             "\\midrule\n")
    for r in rows:
        fh.write(f"{r['mu'] * 100:g} & {r['rho']:g} & {r['p_zero'] * 100:.0f} & "
                 f"{r['cov_naive'] * 100:.0f} & {r['cov_estimated'] * 100:.0f} & "
                 f"{r['cov_oracle'] * 100:.0f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

worst = min(rows, key=lambda r: r["cov_naive"])
macro("SimReps", str(REPS))
macro("WorstNaiveCov", f"{worst['cov_naive'] * 100:.0f}")
macro("WorstNaiveMu", f"{worst['mu'] * 100:g}")
macro("WorstNaiveRho", f"{worst['rho']:g}")
macro("WorstOracleCov", f"{worst['cov_oracle'] * 100:.0f}")

# ---- Estimator sanity check: do both ICC estimators recover rho? --------------
check = []
for rho in (0.05, 0.2, 0.5):
    a_hat, b_hat = [], []
    for _ in range(300):
        mu = 0.1
        a, b = mu * (1 - rho) / rho, (1 - mu) * (1 - rho) / rho
        k = rng.binomial(m, rng.beta(a, b, K))
        a_hat.append(icc_anova(k, np.full(K, m)))
        b_hat.append(icc_betabinom(k, np.full(K, m))[1])
    check.append((rho, np.nanmedian(a_hat), np.nanmedian(b_hat)))
    print(f"rho={rho}: median ANOVA {check[-1][1]:.3f}, median beta-binomial {check[-1][2]:.3f}")
    tag = {0.05: "Five", 0.2: "Twenty", 0.5: "Fifty"}[rho]
    macro(f"IccCheckAnova{tag}", f"{check[-1][1]:.3f}")
    macro(f"IccCheckBB{tag}", f"{check[-1][2]:.3f}")

# Worked example for the reanalysis section: 0 events in 200 runs, naive vs 20 scenarios.
macro("IllusNaive", f"{cp_upper(0, 200) * 100:.1f}")
macro("IllusTwenty", f"{cp_upper(0, 20) * 100:.1f}")

with open(os.path.join(ROOT, "paper", "numbers_sim.tex"), "w") as fh:
    fh.write("% generated by analysis/sim.py; do not edit\n")
    for k_, v in macros.items():
        fh.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
print(macros)
