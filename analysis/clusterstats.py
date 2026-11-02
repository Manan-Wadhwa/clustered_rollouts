"""Cluster-aware rate estimates and zero-event bounds for binary eval outcomes.

Each cluster is one scenario configuration (e.g. scenario x goal x urgency).
Inputs are per-cluster event counts k_i out of m_i rollouts.
"""
import numpy as np
from scipy import optimize, special, stats


def cp_upper(x, n, alpha=0.05):
    """One-sided Clopper-Pearson upper bound. With x = 0 this is 1 - alpha**(1/n),
    which the rule of three approximates as 3/n. Accepts non-integer x and n
    (Kish effective counts) through the beta quantile."""
    if x >= n:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, x + 1, n - x))


def cp_two_sided(x, n, alpha=0.05):
    lo = 0.0 if x <= 0 else float(stats.beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x >= n else float(stats.beta.ppf(1 - alpha / 2, x + 1, n - x))
    return lo, hi


def icc_anova(k, m):
    """One-way ANOVA estimator of the intra-cluster correlation for binary data
    with unequal cluster sizes (Fleiss; Ridout et al. 1999, estimator 'A').
    Returns nan when there are no events or no non-events."""
    k, m = np.asarray(k, float), np.asarray(m, float)
    K, N = len(k), m.sum()
    p = k.sum() / N
    if p in (0.0, 1.0) or K < 2:
        return float("nan")
    pi = k / m
    msb = np.sum(m * (pi - p) ** 2) / (K - 1)
    msw = np.sum(m * pi * (1 - pi)) / (N - K)
    m0 = (N - np.sum(m**2) / N) / (K - 1)
    return float((msb - msw) / (msb + (m0 - 1) * msw))


def _bb_negll(params, k, m):
    mu, rho = special.expit(params)
    a = mu * (1 - rho) / rho
    b = (1 - mu) * (1 - rho) / rho
    ll = special.betaln(k + a, m - k + b) - special.betaln(a, b)
    return -np.sum(ll)


def icc_betabinom(k, m):
    """Beta-binomial maximum-likelihood fit. In the beta-binomial model the
    intra-cluster correlation equals rho = 1 / (a + b + 1).
    Returns (mu, rho); rho is nan when there are no events."""
    k, m = np.asarray(k, float), np.asarray(m, float)
    p = k.sum() / m.sum()
    if p in (0.0, 1.0):
        return float(p), float("nan")
    best = None
    for rho0 in (0.01, 0.1, 0.4):
        x0 = special.logit([np.clip(p, 1e-4, 1 - 1e-4), rho0])
        r = optimize.minimize(_bb_negll, x0, args=(k, m), method="Nelder-Mead",
                              options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
        if best is None or r.fun < best.fun:
            best = r
    mu, rho = special.expit(best.x)
    return float(mu), float(rho)


def deff(m_bar, icc):
    """Kish design effect for equal-size clusters: 1 + (m - 1) * ICC."""
    return 1.0 + (m_bar - 1.0) * icc


def deff_unequal(m, icc):
    """Design effect with unequal cluster sizes, using the size-weighted mean
    m_w = sum(m^2) / sum(m) in place of m."""
    m = np.asarray(m, float)
    return deff(np.sum(m**2) / np.sum(m), icc)


def adjusted_upper(x, n, d, alpha=0.05):
    """Upper bound after shrinking the sample to its effective size n / d.
    Events are scaled the same way, so the point estimate is unchanged."""
    return cp_upper(x / d, n / d, alpha)


def cluster_bootstrap_ci(k, m, B=5000, alpha=0.05, seed=0):
    """Percentile interval for the pooled rate, resampling whole clusters."""
    rng = np.random.default_rng(seed)
    k, m = np.asarray(k, float), np.asarray(m, float)
    idx = rng.integers(0, len(k), size=(B, len(k)))
    rates = k[idx].sum(1) / m[idx].sum(1)
    return tuple(np.quantile(rates, [alpha / 2, 1 - alpha / 2]))


def icc_bootstrap_ci(k, m, B=1000, alpha=0.05, seed=0, estimator=icc_anova):
    """Percentile interval for an ICC estimator, resampling whole clusters.
    Resamples with no events give nan and are dropped; the count is returned."""
    rng = np.random.default_rng(seed)
    k, m = np.asarray(k, float), np.asarray(m, float)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(k), len(k))
        v = estimator(k[i], m[i])
        vals.append(v if np.isscalar(v) else v[1])
    vals = np.asarray(vals)
    ok = vals[~np.isnan(vals)]
    return (*np.quantile(ok, [alpha / 2, 1 - alpha / 2]), int(np.isnan(vals).sum()))
