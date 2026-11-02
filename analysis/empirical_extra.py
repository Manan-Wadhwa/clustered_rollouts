# /// script
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Extra analyses of the real rollouts asked for in review round 1 (B4 to B11, MR F17).

Unit of analysis: a within-behaviour cell, one model and one behaviour (blackmail or
leaking), 15 configurations of 100 rollouts. A cell is "identifiable" when events
occur in at least 2 configurations. Both label sets are used: Opus
(data/rollouts_opus.csv) and the gpt-oss-120b judge (data/rollouts_judge.csv).

What it computes, per identifiable cell and label set:
  B4   ANOVA and beta-binomial ICC against the harmful rate, Spearman correlation.
  B11  ANOVA ICC: percentile cluster bootstrap, BOOT_B resamples, seed BOOT_SEED,
       resamples with no events dropped. Beta-binomial ICC: profile-likelihood
       interval and a parametric-bootstrap percentile interval (PB_B fits).
       The paper's own setting (1,000 resamples, seed 0) is re-run over 10 seeds
       for gpt-oss-20b leaking.
  B9   Kill criterion on the lower limit: DEFF_lo = 1 + 99 * max(ICC_lo, 0).
  B6   Residual ICC after the design factors: logistic model with configuration effect with
       fixed effects for urgency (3 levels) and goal condition (5 levels), fitted
       by maximum likelihood with adaptive Gauss-Hermite quadrature (AGQ_NODES
       nodes per configuration). Also the ANOVA ICC with the control arms
       (goal none, urgency none) dropped.
  B7   Beta goodness of fit: observed against predicted zero-count configurations,
       with a parametric-bootstrap p-value (refit in every replicate).
  B5   Leave-one-model-out ICC transfer on leaking cells, and the zero-event bounds
       of Table 5 (K = 15, m = 100, no events) at every ICC measured here.
  F17  Pooled (30-configuration) cells: why the ANOVA and beta-binomial ICC differ.

Writes paper/numbers_emp.tex, paper/tab_emp_*.tex, paper/figs/icc_vs_rate.pdf and
data/emp_cells.csv.
Run: uv run analysis/empirical_extra.py   (about 7 minutes on a laptop CPU)
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, special, stats

from analyze_real import PRETTY, tex_name
from bounds_extra import any_event_upper, bb_zero_upper
from clusterstats import adjusted_upper, deff, icc_anova, icc_betabinom, icc_bootstrap_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = {"Opus": "rollouts_opus.csv", "Judge": "rollouts_judge.csv"}
BEHAVIOURS = ("blackmail", "leaking")
M = 100            # rollouts per configuration
K_ZERO = 15        # Table 5 setting
BOOT_B, BOOT_SEED = 10000, 1
PB_B, PB_SEED = int(os.getenv("EMP_PB_B", "1000")), 2
AGQ_NODES = 30
BETA_BOUND = 25.0  # bound on fixed effects; hit when an arm has no events (separation)
CHI2_95 = stats.chi2.ppf(0.95, 1)
DEFF_KILL = 1.5
ICC_KILL = (DEFF_KILL - 1) / (M - 1)
GOALS = ["none-none", "explicit-america", "explicit-global", "latent-america", "latent-global"]
URG = ["none", "replacement", "restriction"]


# ---------------------------------------------------------------- data
def load(fname):
    cells = defaultdict(lambda: [0, 0])
    meta = {}
    with open(os.path.join(ROOT, "data", fname)) as fh:
        for r in csv.DictReader(fh):
            key = (r["model"], r["cluster"])
            cells[key][0] += int(r["harmful"])
            cells[key][1] += 1
            meta[r["cluster"]] = (r["scenario"], f"{r['goal_type']}-{r['goal_value']}", r["urgency"])
    out = {}
    for model in {m for m, _ in cells}:
        for beh in BEHAVIOURS + ("all",):
            names = sorted(c for (mm, c) in cells if mm == model and (beh == "all" or meta[c][0] == beh))
            out[(model, beh)] = dict(
                names=names,
                k=np.array([cells[(model, c)][0] for c in names], float),
                m=np.array([cells[(model, c)][1] for c in names], float),
                goal=[meta[c][1] for c in names], urg=[meta[c][2] for c in names])
    return out


# ---------------------------------------------------------------- beta-binomial helpers
def bb_ll(mu, rho, k, m):
    if rho <= 1e-9:  # binomial limit
        return float(np.sum(stats.binom.logpmf(k, m, mu)))
    s = (1 - rho) / rho
    a, b = mu * s, (1 - mu) * s
    return float(np.sum(special.gammaln(m + 1) - special.gammaln(k + 1) - special.gammaln(m - k + 1)
                        + special.betaln(k + a, m - k + b) - special.betaln(a, b)))


def bb_profile(rho, k, m):
    """max over mu of the beta-binomial log-likelihood at fixed rho."""
    r = optimize.minimize_scalar(lambda t: -bb_ll(special.expit(t), rho, k, m),
                                 bounds=(-16, 6), method="bounded", options={"xatol": 1e-8})
    return -r.fun


def bb_profile_ci(k, m, mu_hat, rho_hat):
    """95% profile-likelihood interval for rho. Lower limit 0 when the deviance at
    rho -> 0 (the binomial model) is within the chi-square(1) cut-off."""
    ll_max = bb_ll(mu_hat, rho_hat, k, m)
    grid = np.unique(np.concatenate([np.logspace(-6, np.log10(0.98), 240), [rho_hat]]))
    prof = np.array([bb_profile(r, k, m) for r in grid])
    ll_max = max(ll_max, prof.max())
    dev = 2 * (ll_max - prof)
    ok = dev <= CHI2_95
    i0, i1 = np.argmax(ok), len(ok) - 1 - np.argmax(ok[::-1])

    def cross(i, j):  # interpolate in log rho between grid points i (outside) and j (inside)
        x = np.log([grid[i], grid[j]]); y = dev[[i, j]] - CHI2_95
        return float(np.exp(x[0] + (x[1] - x[0]) * y[0] / (y[0] - y[1])))
    lo = 0.0 if i0 == 0 else cross(i0 - 1, i0)
    hi = 1.0 if i1 == len(grid) - 1 else cross(i1 + 1, i1)
    lr0 = max(2 * (ll_max - bb_ll(k.sum() / m.sum(), 0.0, k, m)), 0.0)
    return lo, hi, lr0


def bb_zero_pred(mu, rho, m):
    """Expected number of configurations with zero events under the fitted Beta."""
    if not rho > 1e-6:  # binomial limit; avoids cancellation in betaln at huge a + b
        return float(np.sum((1 - mu) ** m))
    s = (1 - rho) / rho
    a, b = mu * s, (1 - mu) * s
    return float(np.sum(np.exp(special.betaln(a, b + m) - special.betaln(a, b))))


def bb_cdf_dist(k, mu, rho, mm=M):
    """Cramer-von Mises type distance between the empirical distribution of the
    configuration counts and the fitted beta-binomial (equal sizes mm)."""
    x = np.arange(mm + 1)
    if rho > 1e-6:
        s = (1 - rho) / rho
        pmf = stats.betabinom.pmf(x, mm, mu * s, (1 - mu) * s)
    else:
        pmf = stats.binom.pmf(x, mm, mu)
    F = np.cumsum(pmf)
    Fe = np.array([(k <= xi).mean() for xi in x])
    return float(np.sum((Fe - F) ** 2 * pmf))


def param_boot(k, m, mu, rho, B, seed):
    """Parametric bootstrap from the fitted beta-binomial, refitting each replicate.
    Returns rho* values and the goodness-of-fit statistics of each replicate."""
    rng = np.random.default_rng(seed)
    s = (1 - rho) / rho
    a, b = mu * s, (1 - mu) * s
    rhos, zstat, dstat = [], [], []
    for _ in range(B):
        p = rng.beta(a, b, len(m))
        ks = rng.binomial(m.astype(int), p).astype(float)
        mu_s, rho_s = icc_betabinom(ks, m)
        if rho_s != rho_s:  # no events in the replicate
            rhos.append(np.nan); zstat.append(np.nan); dstat.append(np.nan)
            continue
        rhos.append(rho_s)
        zstat.append((ks == 0).sum() - bb_zero_pred(mu_s, rho_s, m))
        dstat.append(bb_cdf_dist(ks, mu_s, rho_s))
    return np.array(rhos), np.array(zstat), np.array(dstat)


# ---------------------------------------------------------------- logistic model with configuration effect
GH_X, GH_W = np.polynomial.hermite.hermgauss(AGQ_NODES)


def _marg_loglik(eta, logs, k, m):
    """log of prod_i int Bin(k_i | m_i, expit(eta_i + u)) N(u; 0, s^2) du, by adaptive
    Gauss-Hermite quadrature around each configuration's mode. Binomial
    coefficients are left out (constant)."""
    s2 = np.exp(2 * logs)
    u = np.zeros_like(eta)
    for _ in range(300):  # Newton for the mode of the (concave) log integrand, steps clipped
        p = special.expit(eta + u)
        g1 = k - m * p - u / s2
        g2 = -m * p * (1 - p) - 1 / s2
        step = np.clip(g1 / g2, -1.0, 1.0)
        u = u - step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = special.expit(eta + u)
    sd = 1 / np.sqrt(m * p * (1 - p) + 1 / s2)
    uu = u[:, None] + np.sqrt(2) * sd[:, None] * GH_X[None, :]
    lin = eta[:, None] + uu
    g = k[:, None] * -np.logaddexp(0, -lin) + (m - k)[:, None] * -np.logaddexp(0, lin) - uu ** 2 / (2 * s2)
    li = special.logsumexp(g + GH_X[None, :] ** 2, b=GH_W[None, :], axis=1)
    li += np.log(np.sqrt(2) * sd) - 0.5 * np.log(2 * np.pi * s2)
    return float(li.sum())


def design(goal, urg, fixed=True):
    X = [np.ones(len(goal))]
    if fixed:
        X += [np.array([u == lv for u in urg], float) for lv in URG[1:]]
        X += [np.array([g == lv for g in goal], float) for lv in GOALS[1:]]
    return np.column_stack(X)


def fit_glmm(k, m, X):
    """ML fit of logit(p_i) = X_i beta + u_i, u_i ~ N(0, s^2). Returns beta, s2, loglik.
    Also fits s2 = 0 (plain logistic regression) for a boundary likelihood-ratio test."""
    p0 = np.clip(k.sum() / m.sum(), 1e-4, 1 - 1e-4)
    nb = X.shape[1]
    bounds = [(-BETA_BOUND, BETA_BOUND)] * nb
    best = None
    for logs0 in (-3.0, 0.0, 1.0):
        x0 = np.concatenate([[special.logit(p0)], np.zeros(nb - 1), [logs0]])
        r = optimize.minimize(lambda th: -_marg_loglik(X @ th[:nb], th[nb], k, m), x0,
                              method="L-BFGS-B", bounds=bounds + [(np.log(1e-4), np.log(20.0))])
        if best is None or r.fun < best.fun:
            best = r
    beta, logs = best.x[:nb], best.x[nb]

    def nll0(b):
        lin = X @ b
        return -float(np.sum(k * -np.logaddexp(0, -lin) + (m - k) * -np.logaddexp(0, lin)))
    r0 = optimize.minimize(nll0, best.x[:nb], method="L-BFGS-B", bounds=bounds)
    lr = max(2 * (r0.fun - best.fun), 0.0)
    # Overdispersion check on the plain logistic fit: residual deviance and Pearson
    # residuals; df = configurations minus fixed effects.
    p = special.expit(X @ r0.x)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.sum(np.where(k > 0, k * np.log(k / (m * p)), 0)
                         + np.where(m - k > 0, (m - k) * np.log((m - k) / (m * (1 - p))), 0))
        pear = (k - m * p) / np.sqrt(m * p * (1 - p))
    pear = np.where(np.isfinite(pear), pear, 0.0)
    df = len(k) - nb
    # Profile-likelihood upper limit for s (fixed effects re-fitted at each s).
    grid = np.logspace(-2, np.log10(8.0), 36)
    prof, betas = [], []
    for sv in grid:
        r = optimize.minimize(lambda b: -_marg_loglik(X @ b, np.log(sv), k, m), best.x[:nb],
                              method="L-BFGS-B", bounds=bounds)
        prof.append(-r.fun); betas.append(r.x)
    prof = np.array(prof)
    llmax = max(-best.fun, prof.max())
    devp = 2 * (llmax - prof)
    s_hat = np.exp(logs)
    above = [i for i in range(len(grid)) if grid[i] > s_hat and devp[i] > CHI2_95]
    if above:
        i = above[0]
        if i == 0:
            s_hi = grid[0]
        else:
            x = np.log(grid[[i - 1, i]]); y = devp[[i - 1, i]] - CHI2_95
            s_hi = float(np.exp(x[0] + (x[1] - x[0]) * (-y[0]) / (y[1] - y[0]))) if y[0] < 0 else grid[i - 1]
        b_hi = betas[i]
    else:
        s_hi, b_hi = np.inf, betas[-1]
    return dict(beta=beta, s2=float(s_hat ** 2), ll=-best.fun, lr=lr,
                p_lr=0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 1.0,  # 50:50 chi-bar-square mixture
                separated=separated(k, m, X),
                dev=float(dev), df=df, p_dev=float(stats.chi2.sf(dev, df)),
                max_pearson=float(np.max(np.abs(pear))), argmax_pearson=int(np.argmax(np.abs(pear))),
                s2_hi=float(s_hi ** 2), beta_hi=b_hi)


def separated(k, m, X):
    """True when some factor level (a dummy column of X) has no events or only events,
    so its fixed effect has no finite maximum likelihood estimate."""
    if X.shape[1] == 1:
        return False
    nu = len(URG) - 1
    masks = [X[:, j] > 0 for j in range(1, X.shape[1])]
    masks += [X[:, 1:1 + nu].sum(1) == 0, X[:, 1 + nu:].sum(1) == 0]  # reference levels
    return any(k[on].sum() == 0 or k[on].sum() == m[on].sum() for on in masks)


def latent_icc(s2):
    return s2 / (s2 + np.pi ** 2 / 3)


def observed_icc(eta, s2):
    """Observed-scale ICC given the fixed part: sum_i Var_u(p_i) / sum_i [Var_u(p_i) + E_u p_i(1 - p_i)],
    with p_i = expit(eta_i + u), u ~ N(0, s2); Gauss-Hermite with 60 nodes."""
    x, w = np.polynomial.hermite.hermgauss(60)
    p = special.expit(eta[:, None] + np.sqrt(2 * s2) * x[None, :])
    w = w / np.sqrt(np.pi)
    e1 = (p * w).sum(1); e2 = (p ** 2 * w).sum(1)
    var_b = e2 - e1 ** 2
    var_w = e1 - e2
    return float(var_b.sum() / (var_b.sum() + var_w.sum()))


def check_agq():
    """Compare the AGQ marginal likelihood with brute-force integration on a fine grid."""
    k = np.array([0, 3, 40, 1, 0, 16, 0], float); m = np.full(7, 100.0)
    eta = np.array([-6, -3, -0.5, -4, -25.0, -3.9, 2.0])
    out = []
    for logs in (np.log(0.05), 0.0, np.log(3.0), np.log(10.0)):
        s = np.exp(logs)
        u = np.linspace(-12 * s - 30, 12 * s + 30, 400001)
        lin = eta[:, None] + u[None, :]
        g = k[:, None] * -np.logaddexp(0, -lin) + (m - k)[:, None] * -np.logaddexp(0, lin) + stats.norm.logpdf(u, 0, s)[None, :]
        brute = float(np.sum(special.logsumexp(g, axis=1) + np.log(u[1] - u[0])))
        out.append(abs(_marg_loglik(eta, logs, k, m) - brute))
    return max(out)


# ---------------------------------------------------------------- per-cell analysis
def analyse_cell(c):
    k, m = c["k"], c["m"]
    res = dict(events=int(k.sum()), N=int(m.sum()), rate=k.sum() / m.sum(), withev=int((k > 0).sum()))
    res["anova"] = icc_anova(k, m)
    lo, hi, nn = icc_bootstrap_ci(k, m, B=BOOT_B, seed=BOOT_SEED)
    res.update(anova_lo=lo, anova_hi=hi, anova_boot_nan=nn)
    plo, phi, _ = icc_bootstrap_ci(k, m)  # the paper's setting: 1,000 resamples, seed 0
    res.update(anova_lo_paper=plo, anova_hi_paper=phi)
    mu, rho = icc_betabinom(k, m)
    res.update(bb_mu=mu, bb=rho)
    lo, hi, lr0 = bb_profile_ci(k, m, mu, rho)
    res.update(bb_lo=lo, bb_hi=hi, bb_lr0=lr0)
    rs, zs, ds = param_boot(k, m, mu, rho, PB_B, PB_SEED)
    ok = ~np.isnan(rs)
    res.update(bb_pb_lo=float(np.quantile(rs[ok], 0.025)), bb_pb_hi=float(np.quantile(rs[ok], 0.975)),
               pb_nan=int((~ok).sum()))
    res["zeros_obs"] = int((k == 0).sum())
    res["zeros_pred"] = bb_zero_pred(mu, rho, m)
    z_obs = res["zeros_obs"] - res["zeros_pred"]
    d_obs = bb_cdf_dist(k, mu, rho)
    res["gof_p_zero"] = float((np.sum(np.abs(zs[ok]) >= abs(z_obs) - 1e-9) + 1) / (ok.sum() + 1))
    res["gof_p_cdf"] = float((np.sum(ds[ok] >= d_obs - 1e-12) + 1) / (ok.sum() + 1))
    res["deff_lo_anova"] = deff(M, max(res["anova_lo"], 0.0))
    res["deff_lo_bb"] = deff(M, res["bb_lo"])
    res["deff"] = deff(M, max(res["anova"], 0.0))
    # B6: residual ICC
    g0 = fit_glmm(k, m, design(c["goal"], c["urg"], fixed=False))
    g1 = fit_glmm(k, m, design(c["goal"], c["urg"], fixed=True))
    X1 = design(c["goal"], c["urg"], fixed=True)
    res.update(lat_raw=latent_icc(g0["s2"]), lat_res=latent_icc(g1["s2"]), s2_raw=g0["s2"], s2_res=g1["s2"],
               obs_raw=observed_icc(np.full(len(k), g0["beta"][0]), g0["s2"]),
               obs_res=observed_icc(X1 @ g1["beta"], g1["s2"]), p_res=g1["p_lr"], sep=g1["separated"],
               lat_res_hi=latent_icc(g1["s2_hi"]), obs_res_hi=observed_icc(X1 @ g1["beta_hi"], g1["s2_hi"]),
               lat_raw_hi=latent_icc(g0["s2_hi"]),
               glm_dev=g1["dev"], glm_df=g1["df"], glm_p_dev=g1["p_dev"], max_pearson=g1["max_pearson"],
               worst_config=c["names"][g1["argmax_pearson"]])
    keep = [i for i, (g, u) in enumerate(zip(c["goal"], c["urg"])) if g != "none-none" and u != "none"]
    res["anova_nocontrol"] = icc_anova(k[keep], m[keep])
    res["k_nocontrol"] = int(k[keep].sum())
    # B5: zero-event bounds at this cell's ICCs
    for est in ("anova", "bb"):
        r = max(res[est], 0.0)
        res[f"zero_exact_{est}"] = bb_zero_upper(K_ZERO, M, r) if r > 0 else 1 - 0.05 ** (1 / (K_ZERO * M))
        res[f"zero_kish_{est}"] = adjusted_upper(0, K_ZERO * M, deff(M, r))
    return res


# ---------------------------------------------------------------- output helpers
def f2(v):
    return "--" if v != v else f"{v:.2f}"


def f3(v):
    return "--" if v != v else f"{v:.3f}"


def pct(v, d=2):
    return f"{v * 100:.{d}f}"


def main():
    print("AGQ check, max |AGQ - brute force| log-likelihood:", check_agq())
    data = {lab: load(f) for lab, f in LABELS.items()}
    models = [m for m in PRETTY if (m, "leaking") in data["Opus"]]
    macros = {"EmpBootB": f"{BOOT_B:,}".replace(",", "{,}"), "EmpParamBootB": f"{PB_B:,}".replace(",", "{,}"),
              "EmpAgqNodes": str(AGQ_NODES), "EmpIccKill": f"{ICC_KILL:.4f}"}
    rows = []
    for lab in LABELS:
        for model in models:
            for beh in BEHAVIOURS:
                c = data[lab][(model, beh)]
                if int((c["k"] > 0).sum()) < 2:
                    continue
                r = analyse_cell(c)
                r.update(labels=lab, model=model, behaviour=beh)
                rows.append(r)
                print(lab, model, beh, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()
                                        if a not in ("labels", "model", "behaviour")})

    # data file with everything
    keys = list(rows[0])
    with open(os.path.join(ROOT, "data", "emp_cells.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def cellname(r):
        return f"{r['labels']}{tex_name(r['model'])}{r['behaviour'].title()}"

    for r in rows:
        n = "Emp" + cellname(r)
        macros.update({
            n + "IccAnovaLo": f2(max(r["anova_lo"], 0)), n + "IccAnovaHi": f2(r["anova_hi"]),
            n + "IccBB": f2(r["bb"]), n + "IccBBLo": f3(r["bb_lo"]), n + "IccBBHi": f2(r["bb_hi"]),
            n + "IccBBPbLo": f3(r["bb_pb_lo"]), n + "IccBBPbHi": f2(r["bb_pb_hi"]),
            n + "DeffLoAnova": f"{r['deff_lo_anova']:.1f}", n + "DeffLoBB": f"{r['deff_lo_bb']:.1f}",
            n + "LatentRaw": f2(r["lat_raw"]), n + "LatentResid": f2(r["lat_res"]),
            n + "ObsResid": f2(r["obs_res"]), n + "ObsRaw": f2(r["obs_raw"]),
            n + "ResidP": f"{r['p_res']:.3f}", n + "IccNoControl": f2(max(r["anova_nocontrol"], 0)),
            n + "LatentResidHi": f2(r["lat_res_hi"]), n + "ObsResidHi": f3(r["obs_res_hi"]),
            n + "GlmDev": f"{r['glm_dev']:.1f}", n + "GlmDevP": f"{r['glm_p_dev']:.3f}",
            n + "ZerosObs": str(r["zeros_obs"]), n + "ZerosPred": f"{r['zeros_pred']:.1f}",
            n + "GofP": f"{r['gof_p_zero']:.2f}", n + "GofCdfP": f"{r['gof_p_cdf']:.2f}",
        })

    # ---- B4: ICC against rate
    for lab in LABELS:
        sub = [r for r in rows if r["labels"] == lab]
        macros[f"EmpCells{lab}"] = str(len(sub))
        for est, key in (("Anova", "anova"), ("BB", "bb"), ("Latent", "lat_raw")):
            rs = stats.spearmanr([r["rate"] for r in sub], [r[key] for r in sub])
            macros[f"EmpSpearman{est}{lab}"] = f"{rs.statistic:.2f}"
            macros[f"EmpSpearman{est}{lab}P"] = f"{rs.pvalue:.2f}"
            print("Spearman", lab, est, rs.statistic, rs.pvalue, "n", len(sub))
        for beh in BEHAVIOURS:
            sb = [r for r in sub if r["behaviour"] == beh]
            if len(sb) >= 3:
                rs = stats.spearmanr([r["rate"] for r in sb], [r["anova"] for r in sb])
                macros[f"EmpSpearmanAnova{lab}{beh.title()}"] = f"{rs.statistic:.2f}"
                print("Spearman within", lab, beh, rs.statistic, rs.pvalue, len(sb))
    plot_icc_rate(rows)

    # ---- B9: kill criterion on the lower limit
    for lab in LABELS:
        sub = [r for r in rows if r["labels"] == lab]
        macros[f"EmpKillPoint{lab}"] = str(sum(r["deff"] > DEFF_KILL for r in sub))
        macros[f"EmpKillLoAnova{lab}"] = str(sum(r["deff_lo_anova"] > DEFF_KILL for r in sub))
        macros[f"EmpKillLoBB{lab}"] = str(sum(r["deff_lo_bb"] > DEFF_KILL for r in sub))
        macros[f"EmpFewConfigs{lab}"] = str(sum(r["withev"] < 5 for r in sub))

    # ---- B11: stability of the paper's gpt-oss-20b leaking interval
    c = data["Opus"][("gpt_oss_20b", "leaking")]
    los = [icc_bootstrap_ci(c["k"], c["m"], B=1000, seed=s)[0] for s in range(10)]
    print("gpt-oss-20b leaking, paper setting, lower limit over seeds 0-9:", np.round(los, 4))
    macros["EmpGptLeakPaperLoSeedZero"] = f2(max(los[0], 0))
    macros["EmpGptLeakPaperLoMin"] = f2(max(min(los), 0))
    macros["EmpGptLeakPaperLoMax"] = f2(max(los))
    macros["EmpGptLeakPaperLoZeroSeeds"] = str(sum(round(max(v, 0), 2) == 0 for v in los))

    # ---- B6 summary
    for lab in LABELS:
        sub = [r for r in rows if r["labels"] == lab]
        ratio = [r["lat_res"] / r["lat_raw"] for r in sub if r["lat_raw"] > 0]
        macros[f"EmpResidShareMin{lab}"] = f"{min(ratio) * 100:.0f}"
        macros[f"EmpResidShareMax{lab}"] = f"{max(ratio) * 100:.0f}"
        macros[f"EmpResidLatentMin{lab}"] = f2(min(r["lat_res"] for r in sub))
        macros[f"EmpResidLatentMax{lab}"] = f2(max(r["lat_res"] for r in sub))
        macros[f"EmpResidSig{lab}"] = str(sum(r["p_res"] < 0.05 for r in sub))
        macros[f"EmpResidZero{lab}"] = str(sum(r["lat_res"] < 0.001 for r in sub))
        macros[f"EmpResidLatentHiMax{lab}"] = f2(max(r["lat_res_hi"] for r in sub))
        macros[f"EmpResidObsHiMax{lab}"] = f3(max(r["obs_res_hi"] for r in sub))
        macros[f"EmpGlmOverdisp{lab}"] = str(sum(r["glm_p_dev"] < 0.05 for r in sub))
        macros[f"EmpSeparated{lab}"] = str(sum(r["sep"] for r in sub))
        macros[f"EmpGofReject{lab}"] = str(sum(min(r["gof_p_zero"], r["gof_p_cdf"]) < 0.05 for r in sub))

    # ---- B5: zero-event sensitivity over measured ICCs, and leave-one-model-out
    allrho = [(max(r[e], 0.0), r, e) for r in rows for e in ("anova", "bb")]
    lo_r = min(allrho, key=lambda t: t[0]); hi_r = max(allrho, key=lambda t: t[0])
    macros.update({
        "EmpZeroRhoMin": f3(lo_r[0]), "EmpZeroRhoMax": f2(hi_r[0]),
        "EmpZeroExactMin": pct(bb_zero_upper(K_ZERO, M, lo_r[0])), "EmpZeroExactMax": pct(bb_zero_upper(K_ZERO, M, hi_r[0])),
        "EmpZeroKishMin": pct(adjusted_upper(0, K_ZERO * M, deff(M, lo_r[0]))),
        "EmpZeroKishMax": pct(adjusted_upper(0, K_ZERO * M, deff(M, hi_r[0]))),
        "EmpZeroScen": pct(any_event_upper(0, K_ZERO), 1),
        "EmpZeroNaive": pct(1 - 0.05 ** (1 / (K_ZERO * M))),
        "EmpZeroRhoMinCell": f"{PRETTY[lo_r[1]['model']]} {lo_r[1]['behaviour']}",
        "EmpZeroRhoMaxCell": f"{PRETTY[hi_r[1]['model']]} {hi_r[1]['behaviour']}",
    })
    print("zero-event range", lo_r[0], hi_r[0], macros["EmpZeroExactMin"], macros["EmpZeroExactMax"])
    lomo = leave_one_out(rows, models)
    for lab in LABELS:
        for est in ("anova", "bb"):
            rr = [x["ratio"] for x in lomo if x["labels"] == lab and x["est"] == est]
            br = [x["bound_ratio"] for x in lomo if x["labels"] == lab and x["est"] == est]
            E = "Anova" if est == "anova" else "BB"
            macros[f"EmpLomoRatioMin{E}{lab}"] = f2(min(rr))
            macros[f"EmpLomoRatioMax{E}{lab}"] = f2(max(rr))
            macros[f"EmpLomoBoundRatioMin{E}{lab}"] = f2(min(br))
            macros[f"EmpLomoBoundRatioMax{E}{lab}"] = f2(max(br))
    # blackmail: the paper's own transfer, Mistral-Small-24B to Qwen3-32B
    for lab in LABELS:
        d = {r["model"]: r for r in rows if r["labels"] == lab and r["behaviour"] == "blackmail"}
        if "mistral_small_24b" in d and "qwen3_32b" in d:
            macros[f"EmpBlackmailTransfer{lab}"] = f2(d["qwen3_32b"]["anova"] / d["mistral_small_24b"]["anova"])

    # ---- B15: configurations outside the binomial-noise range of Figure 4
    for lab in LABELS:
        out = tot = 0
        for model in models:
            for beh in BEHAVIOURS:
                c = data[lab][(model, beh)]
                p = c["k"].sum() / c["m"].sum()
                lo, hi = stats.binom.ppf([0.025, 0.975], M, p)
                out += int(np.sum((c["k"] < lo) | (c["k"] > hi)))
                tot += len(c["k"])
        macros[f"EmpOutsideBand{lab}"] = str(out)
        macros[f"EmpConfigsTotal{lab}"] = str(tot)
        print("outside binomial band", lab, out, "of", tot)

    # ---- F17: pooled cells
    for model in models:
        c = data["Opus"][(model, "all")]
        mu, rho = icc_betabinom(c["k"], c["m"])
        n = "EmpPooled" + tex_name(model)
        s = (1 - rho) / rho
        macros[n + "BetaA"] = f2(mu * s)
        macros[n + "ZerosObs"] = str(int((c["k"] == 0).sum()))
        macros[n + "ZerosPred"] = f"{bb_zero_pred(mu, rho, c['m']):.1f}"
        macros[n + "MuBB"] = pct(mu, 1)
        print("pooled", model, "anova", round(icc_anova(c["k"], c["m"]), 3), "bb", round(rho, 3), "a", round(mu * s, 3),
              "b", round((1 - mu) * s, 2), "mu_bb", round(mu, 4), "rate", round(c["k"].sum() / c["m"].sum(), 4),
              "zeros", int((c["k"] == 0).sum()), "pred", round(bb_zero_pred(mu, rho, c["m"]), 2))

    write_tables(rows, lomo)
    with open(os.path.join(ROOT, "paper", "numbers_emp.tex"), "w") as fh:
        fh.write("% generated by analysis/empirical_extra.py; do not edit\n")
        for k, v in macros.items():
            assert k.isalpha(), k
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(len(macros), "macros written")


def leave_one_out(rows, models):
    out = []
    for lab in LABELS:
        leak = {r["model"]: r for r in rows if r["labels"] == lab and r["behaviour"] == "leaking"}
        for est in ("anova", "bb"):
            for mo, r in leak.items():
                others = [max(leak[o][est], 0) for o in leak if o != mo]
                borrowed = float(np.median(others))
                true = max(r[est], 0)
                out.append(dict(labels=lab, est=est, model=mo, true=true, borrowed=borrowed,
                                ratio=true / borrowed, lo=min(others), hi=max(others),
                                bound_true=bb_zero_upper(K_ZERO, M, true),
                                bound_borrowed=bb_zero_upper(K_ZERO, M, borrowed),
                                bound_ratio=bb_zero_upper(K_ZERO, M, borrowed) / bb_zero_upper(K_ZERO, M, true)))
    return out


def plot_icc_rate(rows):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    col = {"blackmail": "#0072B2", "leaking": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    for ax, (est, lo, hi, title) in zip(axes, (("anova", "anova_lo", "anova_hi", "ANOVA ICC, 95% resampling interval"),
                                               ("bb", "bb_lo", "bb_hi", "beta-binomial ICC, 95% likelihood interval"))):
        for r in rows:
            x = r["rate"] * 100 * (1.04 if r["labels"] == "Judge" else 1 / 1.04)  # small offset so pairs do not overlap
            y = max(r[est], 0)
            yl, yh = max(r[lo], 0), r[hi]
            filled = r["labels"] == "Opus"
            ax.errorbar(x, y, yerr=[[y - yl], [yh - y]], fmt="none", ecolor=col[r["behaviour"]], elinewidth=0.8, alpha=0.6)
            ax.plot(x, y, "o" if filled else "s", ms=5, mfc=col[r["behaviour"]] if filled else "white",
                    mec=col[r["behaviour"]], mew=1.2)
        ax.set_xscale("log")
        ticks = [0.5, 1, 2, 5, 10, 20, 50]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(0.45, 90)
        ax.set_xlabel("harmful rate in the cell (%), log scale")
        ax.set_title(title, fontsize=8)
        ax.axhline(ICC_KILL, color="0.5", lw=0.8, ls=":")
        ax.grid(axis="y", color="0.9", lw=0.5)
    axes[0].set_ylabel("ICC")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=col["blackmail"], label="blackmail"),
         Line2D([], [], marker="o", ls="", color=col["leaking"], label="leaking"),
         Line2D([], [], marker="o", ls="", mfc="0.3", mec="0.3", label="Opus labels"),
         Line2D([], [], marker="s", ls="", mfc="white", mec="0.3", label="judge labels"),
         Line2D([], [], color="0.5", lw=0.8, ls=":", label=f"DEFF = 1.5 (ICC = {ICC_KILL:.4f})")]
    fig.legend(handles=h, frameon=False, loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(ROOT, "paper", "figs", "icc_vs_rate.pdf"))


def write_tables(rows, lomo):
    P = os.path.join(ROOT, "paper")
    hdr = "% generated by analysis/empirical_extra.py; do not edit\n"
    order = sorted(rows, key=lambda r: (r["labels"] != "Opus", r["behaviour"], r["rate"]))

    def name(r):
        dag = "$^\\dagger$" if r["withev"] < 5 else ""
        return f"{PRETTY[r['model']]}{dag}"

    # cells: ICC, intervals, kill criterion on the lower limit
    with open(os.path.join(P, "tab_emp_cells.tex"), "w") as fh:
        fh.write(hdr + "\\begin{tabular}{lllrrllllrr}\n\\toprule\n"
                 "labels & behaviour & model & rate (\\%) & configs & ICC ANOVA [resampling] & ICC beta-bin. & "
                 "[profile] & [param.\\ boot.] & \\multicolumn{2}{c}{DEFF at lower limit} \\\\\n"
                 "& & & & with events & & & & & ANOVA & beta-bin. \\\\\n\\midrule\n")
        prev = None
        for r in order:
            if prev and (r["labels"], r["behaviour"]) != prev:
                fh.write("\\addlinespace\n")
            prev = (r["labels"], r["behaviour"])
            fh.write(f"{r['labels']} & {r['behaviour']} & {name(r)} & {pct(r['rate'], 1)} & {r['withev']} & "
                     f"{f2(max(r['anova'], 0))} [{f2(max(r['anova_lo'], 0))}, {f2(r['anova_hi'])}] & {f2(r['bb'])} & "
                     f"[{f3(r['bb_lo'])}, {f2(r['bb_hi'])}] & [{f3(r['bb_pb_lo'])}, {f2(r['bb_pb_hi'])}] & "
                     f"{r['deff_lo_anova']:.1f} & {r['deff_lo_bb']:.1f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # residual ICC and Beta goodness of fit
    with open(os.path.join(P, "tab_emp_resid.tex"), "w") as fh:
        fh.write(hdr + "\\begin{tabular}{lllrrrrrrrrrr}\n\\toprule\n"
                 "& & & \\multicolumn{2}{c}{ANOVA ICC} & \\multicolumn{5}{c}{logistic model with configuration effect} & "
                 "\\multicolumn{2}{c}{zero configs} & Beta \\\\\n"
                 "\\cmidrule(lr){4-5}\\cmidrule(lr){6-10}\\cmidrule(lr){11-12}\n"
                 "labels & behaviour & model & all 15 & no controls (8) & latent, raw & latent, resid.\\ & "
                 "observed, resid.\\ & dev.\\ (8 df) & $p$ & obs.\\ & pred.\\ & GOF $p$ \\\\\n\\midrule\n")
        prev = None
        for r in order:
            if prev and (r["labels"], r["behaviour"]) != prev:
                fh.write("\\addlinespace\n")
            prev = (r["labels"], r["behaviour"])
            sep = "$^\\ast$" if r["sep"] else ""
            fh.write(f"{r['labels']} & {r['behaviour']} & {name(r)}{sep} & {f2(max(r['anova'], 0))} & "
                     f"{f2(max(r['anova_nocontrol'], 0))} & {f2(r['lat_raw'])} & {f2(r['lat_res'])} [{f2(r['lat_res_hi'])}] & "
                     f"{f3(r['obs_res'])} [{f3(r['obs_res_hi'])}] & {r['glm_dev']:.1f} & {r['glm_p_dev']:.3f} & "
                     f"{r['zeros_obs']} & {r['zeros_pred']:.1f} & {min(r['gof_p_zero'], r['gof_p_cdf']):.2f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # zero-event bounds over measured ICCs
    with open(os.path.join(P, "tab_emp_zero.tex"), "w") as fh:
        fh.write(hdr + "\\begin{tabular}{lllrrrrrr}\n\\toprule\n"
                 "& & & \\multicolumn{3}{c}{ANOVA ICC} & \\multicolumn{3}{c}{beta-binomial ICC} \\\\\n"
                 "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
                 "labels & behaviour & model & ICC & beta-bin. (\\%) & des.\\ eff. (\\%) & ICC & beta-bin. (\\%) & des.\\ eff. (\\%) \\\\\n"
                 "\\midrule\n")
        for r in sorted(rows, key=lambda r: (r["labels"] != "Opus", r["behaviour"], r["anova"])):
            fh.write(f"{r['labels']} & {r['behaviour']} & {PRETTY[r['model']]} & {f3(max(r['anova'], 0))} & "
                     f"{pct(r['zero_exact_anova'])} & {pct(r['zero_kish_anova'])} & {f3(r['bb'])} & "
                     f"{pct(r['zero_exact_bb'])} & {pct(r['zero_kish_bb'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # leave-one-model-out on leaking
    with open(os.path.join(P, "tab_emp_lomo.tex"), "w") as fh:
        fh.write(hdr + "\\begin{tabular}{lllrrrrr}\n\\toprule\n"
                 "labels & estimator & model & true ICC & borrowed ICC & others' range & true / borrowed & "
                 "bound ratio \\\\\n\\midrule\n")
        prev = None
        for x in lomo:
            if prev and (x["labels"], x["est"]) != prev:
                fh.write("\\addlinespace\n")
            prev = (x["labels"], x["est"])
            fh.write(f"{x['labels']} & {'ANOVA' if x['est'] == 'anova' else 'beta-bin.'} & {PRETTY[x['model']]} & "
                     f"{f3(x['true'])} & {f3(x['borrowed'])} & {f3(x['lo'])}--{f3(x['hi'])} & {f2(x['ratio'])} & "
                     f"{f2(x['bound_ratio'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
