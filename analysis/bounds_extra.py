"""Upper bounds for a harmful-action rate when rollouts are clustered by scenario.

Model: scenario i has its own rate p_i ~ Beta(a, b) with mean mu and
intra-cluster correlation rho = 1 / (a + b + 1); its count is
k_i ~ Binomial(m_i, p_i). The target is mu, the scenario-average rate.

The main function is `bb_upper`, the exact one-sided upper bound for mu from
the total count X = sum k_i with rho held fixed at a supplied value:

    UB(x) = sup { mu : P(X <= x | mu, rho, m_1..m_K) >= alpha }.

With x = 0 this is the largest mu with prod_i B(a, b + m_i) / B(a, b) >= alpha,
the beta-binomial version of the Clopper-Pearson bound 1 - alpha**(1/N).
At rho = 0 it equals Clopper-Pearson on N = sum m_i trials; at rho = 1 every
scenario is all-or-nothing and the zero-event bound is 1 - alpha**(1/K).

Validity: for fixed rho, Beta(mu s, (1 - mu) s) with s = (1 - rho)/rho is
ordered in mu by likelihood ratio, so each k_i, and hence X, is stochastically
increasing in mu. Inverting the one-sided test therefore gives coverage of at
least 1 - alpha whenever the assumed rho is the true one and the p_i really are
Beta. Both assumptions matter; see sim_extra.py for what happens when they fail.

Cluster-level alternatives. A t interval or a cluster bootstrap on the K
scenario means degenerates with zero events (every mean is 0, the spread is 0).
The bound that survives is `any_event_upper`: count the scenarios with at least
one event and apply Clopper-Pearson to that count out of K. It needs no rho and
no Beta assumption, only independent scenarios, but it is conservative.
"""
import numpy as np
from scipy import optimize, special, stats


def _s(rho):
    """Beta precision a + b = (1 - rho) / rho, with the limits 0 and inf."""
    if rho <= 0:
        return np.inf
    if rho >= 1:
        return 0.0
    return (1.0 - rho) / rho


def beta_params(mu, rho):
    """(a, b) of the Beta distribution with mean mu and ICC rho (0 < rho < 1)."""
    s = _s(rho)
    return mu * s, (1.0 - mu) * s


def effective_trials(m, rho):
    """Effective number of independent trials one cluster of size m contributes
    to a rare-event bound: sum_{j=0}^{m-1} s / (s + j), s = (1 - rho)/rho.

    As mu -> 0, log P(cluster has 0 events) = -mu * effective_trials(m, rho)
    + O(mu^2), so the zero-event bound is about 1 - alpha**(1 / sum_i m_eff_i).
    Equals m at rho = 0 and 1 at rho = 1. For 0 < rho < 1 it is about
    s * log(1 + m / s), which keeps growing with m; the Kish effective size
    m / (1 + (m - 1) rho) stops growing at 1 / rho."""
    s = _s(rho)
    if np.isinf(s):
        return float(m)
    if s == 0:
        return 1.0
    j = np.arange(int(m))
    return float(np.sum(s / (s + j)))


def log_p_zero_cluster(mu, m, rho):
    """log P(k = 0) for one cluster of size m: log B(a, b + m) - log B(a, b).

    Uses the product form B(a, b + m) / B(a, b) = prod_{j<m} (b + j)/(a + b + j)
    = prod_{j<m} (1 - mu s / (s + j)), which stays accurate at both rho -> 0
    (each factor -> 1 - mu) and rho -> 1 (only the j = 0 factor, 1 - mu, is left).
    """
    s = _s(rho)
    if np.isinf(s):
        return m * np.log1p(-mu)
    if s == 0:
        return np.log1p(-mu)
    j = np.arange(int(m))
    return float(np.sum(np.log1p(-mu * s / (s + j))))


def log_p_zero(mu, sizes, rho):
    """log P(X = 0) over clusters with sizes m_1..m_K (any order)."""
    vals, counts = np.unique(np.asarray(sizes, int), return_counts=True)
    return sum(c * log_p_zero_cluster(mu, v, rho) for v, c in zip(vals, counts))


def cluster_pmf(m, mu, rho):
    """pmf of k in 0..m for one cluster: beta-binomial, or its rho = 0 / 1 limits."""
    k = np.arange(m + 1)
    if rho <= 0:
        return stats.binom.pmf(k, m, mu)
    if rho >= 1:
        out = np.zeros(m + 1)
        out[0], out[m] = 1 - mu, mu
        return out
    a, b = beta_params(mu, rho)
    return stats.betabinom.pmf(k, m, a, b)


def _conv_power(p, n, xmax):
    """n-fold self-convolution of pmf p, keeping only entries 0..xmax.
    Truncation is exact for those entries because pmfs live on k >= 0."""
    out, base = np.array([1.0]), p[: xmax + 1]
    while n:
        if n & 1:
            out = np.convolve(out, base)[: xmax + 1]
        n >>= 1
        if n:
            base = np.convolve(base, base)[: xmax + 1]
    return out


def sum_pmf(sizes, mu, rho, xmax=None):
    """pmf of X = sum_i k_i on 0..xmax (default: 0..N), clusters independent."""
    sizes = np.asarray(sizes, int)
    if xmax is None:
        xmax = int(sizes.sum())
    out = np.array([1.0])
    for v, c in zip(*np.unique(sizes, return_counts=True)):
        out = np.convolve(out, _conv_power(cluster_pmf(int(v), mu, rho), int(c), xmax))[: xmax + 1]
    return np.pad(out, (0, xmax + 1 - len(out)))


def sum_cdf(x, sizes, mu, rho):
    """P(X <= x | mu, rho)."""
    return float(np.clip(sum_pmf(sizes, mu, rho, xmax=int(x)).sum(), 0.0, 1.0))


def bb_upper(x, sizes, rho, alpha=0.05):
    """Exact one-sided (1 - alpha) upper bound for mu from total count x, with
    the ICC fixed at rho. `sizes` lists the rollouts per scenario.

    x = 0 uses the closed form P(X = 0) = prod_i B(a, b + m_i) / B(a, b).
    Reduces to Clopper-Pearson at rho = 0 and to 1 - alpha**(1/K) at rho = 1
    with x = 0."""
    sizes = np.asarray(sizes, int)
    if x >= sizes.sum():
        return 1.0
    la = np.log(alpha)
    if x == 0:
        f = lambda mu: log_p_zero(mu, sizes, rho) - la
    else:
        f = lambda mu: np.log(max(sum_cdf(x, sizes, mu, rho), 1e-300)) - la
    return float(optimize.brentq(f, 1e-14, 1 - 1e-14, xtol=1e-14, rtol=1e-12))


def bb_zero_upper(K, m, rho, alpha=0.05):
    """Zero-event bound for K scenarios of m rollouts each (convenience wrapper)."""
    return bb_upper(0, np.full(K, m), rho, alpha)


def any_event_upper(y, K, alpha=0.05):
    """Upper bound for mu from y = number of scenarios (out of K) with at least one
    event. Valid for any distribution of scenario rates, because
    P(scenario has an event) = 1 - E[(1 - p)^m] >= 1 - E[1 - p] = mu.
    With y = 0 it is 1 - alpha**(1/K), the rho = 1 limit of `bb_upper`."""
    if y >= K:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, y + 1, K - y))


# ---- Worst case over all rate distributions with a given mean and ICC ------------
#
# Fix the mean mu and the ICC rho, so E[p] = mu and E[p^2] = mu^2 + rho mu (1 - mu),
# and let the scenario rates p_i follow ANY distribution on [0, 1] with those two
# moments. P(scenario has no event) = E[g(p)] with g(p) = (1 - p)^m. Its largest value
# over all such distributions comes from the two-point distribution with an atom at 0:
# p = c with probability pi, else 0, where c = mu + rho (1 - mu) and pi = mu / c.
#
# Proof. Let q be the quadratic with q(0) = g(0), q(c) = g(c), q'(c) = g'(c). By the
# Hermite interpolation error formula, g(p) - q(p) = g'''(xi) / 6 * p (p - c)^2 for some
# xi in [0, 1]. Since g'''(p) = -m (m-1) (m-2) (1-p)^(m-3) <= 0 and p (p - c)^2 >= 0 on
# [0, 1], g <= q there. So for any admissible F, E_F[g] <= E_F[q] = E_2pt[q] = E_2pt[g]:
# q is quadratic, so E[q] depends only on the two fixed moments, and g = q on the
# support {0, c} of the two-point law. The same two-point law maximises every factor of
# P(X = 0) = prod_i E[(1 - p)^(m_i)], so it is the worst case for unequal sizes too.
# A linear program over distributions on a fine grid of p agrees to 1e-7.
#
# So `wc_upper(0, ...)` is the largest mu any rate distribution with ICC rho can have
# while still giving P(X = 0) >= alpha: a zero-event bound that needs rho but not the
# Beta shape. For x > 0 we have no worst-case result. Inverting the two-point model
# alone under-covers when the truth is Beta and events are common (92.9% at rate 5%,
# ICC 0.5, 30 x 100), so for x > 0 `wc_upper` takes the larger of the two-point and
# Beta bounds. sim_robust.py checks its coverage under four shapes.


def twopoint_params(mu, rho):
    """(c, pi) of the two-point law p = c w.p. pi, else 0, with mean mu and ICC rho."""
    c = mu + rho * (1.0 - mu)
    return c, mu / c


def log_p_zero_cluster_twopoint(mu, m, rho):
    """log P(k = 0) for one cluster of size m under the two-point law: the largest
    value any rate distribution with mean mu and ICC rho can give."""
    c, pi = twopoint_params(mu, rho)
    if c >= 1.0:
        return np.log1p(-pi)
    with np.errstate(divide="ignore"):  # P(k = 0) underflows to 0 only as mu -> 1
        return float(np.log1p(-pi * -np.expm1(m * np.log1p(-c))))


def cluster_pmf_twopoint(m, mu, rho):
    """pmf of k in 0..m for one cluster under the two-point law."""
    c, pi = twopoint_params(mu, rho)
    out = pi * stats.binom.pmf(np.arange(m + 1), m, min(c, 1.0))
    out[0] += 1.0 - pi
    return out


def sum_pmf_iid(cluster, K, xmax):
    """pmf of the total over K independent clusters with the same pmf, on 0..xmax."""
    out = _conv_power(np.asarray(cluster, float), int(K), int(xmax))
    return np.pad(out, (0, xmax + 1 - len(out)))


def twopoint_upper(x, sizes, rho, alpha=0.05):
    """One-sided (1 - alpha) upper bound for mu from total count x under the two-point
    model. P(X <= x) is decreasing in mu because both c and pi increase with mu."""
    sizes = np.asarray(sizes, int)
    if x >= sizes.sum():
        return 1.0
    la = np.log(alpha)
    vals, counts = np.unique(sizes, return_counts=True)
    if x == 0:
        f = lambda mu: sum(n * log_p_zero_cluster_twopoint(mu, v, rho)
                           for v, n in zip(vals, counts)) - la
    else:
        def f(mu):
            out = np.array([1.0])
            for v, n in zip(vals, counts):
                out = np.convolve(out, _conv_power(cluster_pmf_twopoint(int(v), mu, rho),
                                                   int(n), int(x)))[: int(x) + 1]
            return np.log(max(float(out.sum()), 1e-300)) - la
    return float(optimize.brentq(f, 1e-14, 1 - 1e-14, xtol=1e-14, rtol=1e-12))


def wc_upper(x, sizes, rho, alpha=0.05):
    """Worst-case bound (see the comment above). x = 0: the largest mu that any rate
    distribution with ICC rho allows, which is the two-point bound. x > 0: the larger
    of the two-point and Beta bounds, a heuristic checked in sim_robust.py."""
    tp = twopoint_upper(x, sizes, rho, alpha)
    return tp if x == 0 else max(tp, bb_upper(x, sizes, rho, alpha))


def wc_zero_upper(K, m, rho, alpha=0.05):
    """Worst-case zero-event bound for K scenarios of m rollouts each."""
    return wc_upper(0, np.full(K, m), rho, alpha)


# ---- Other non-Beta rate distributions with a given mean and ICC -----------------

_Z = np.linspace(-12.0, 12.0, 6001)
_WZ = stats.norm.pdf(_Z)
_WZ = _WZ / _WZ.sum()


def logitnormal_params(mu, rho):
    """(b, s) with logit p ~ Normal(b, s^2) and E[p] = mu, Var(p) = rho mu (1 - mu).
    Moments by summation over a fine grid of normal quantiles."""
    target = mu * mu + rho * mu * (1.0 - mu)

    def b_for(s):
        return optimize.brentq(lambda b: _WZ @ special.expit(b + s * _Z) - mu, -60, 60, xtol=1e-13)

    def gap(log_s):
        s = np.exp(log_s)
        p = special.expit(b_for(s) + s * _Z)
        return np.log(_WZ @ p ** 2) - np.log(target)

    log_s = optimize.brentq(gap, np.log(1e-3), np.log(15.0), xtol=1e-12)
    s = float(np.exp(log_s))
    return b_for(s), s


def cluster_pmf_logitnormal(m, mu, rho):
    """pmf of k in 0..m for one cluster when logit p is normal with mean mu, ICC rho."""
    b, s = logitnormal_params(mu, rho)
    p = special.expit(b + s * _Z)
    k = np.arange(m + 1)
    return _WZ @ stats.binom.pmf(k[None, :], m, p[:, None])


def zib_params(mu, rho, share=0.5):
    """Zero-inflated Beta with mean mu and ICC rho: a fraction 1 - pi of scenarios
    never produce the behaviour (p = 0) and the rest have p ~ Beta with mean nu and
    ICC r. The immune fraction is `share` times that of the two-point law, so
    share = 0 is the Beta and share = 1 the two-point. Returns (pi, nu, r)."""
    _, pi_tp = twopoint_params(mu, rho)
    pi = 1.0 - share * (1.0 - pi_tp)
    nu = mu / pi
    r = ((mu * mu + rho * mu * (1.0 - mu)) / pi - nu * nu) / (nu * (1.0 - nu))
    return pi, nu, r


def cluster_pmf_zib(m, mu, rho, share=0.5):
    """pmf of k in 0..m for one cluster under the zero-inflated Beta."""
    pi, nu, r = zib_params(mu, rho, share)
    out = pi * cluster_pmf(m, nu, r)
    out[0] += 1.0 - pi
    return out
