"""Macros for numbers the prose quotes that no other script writes.

Reads data/sim_extra_unequal.csv, data/sim_extra_budget.csv, data/reanalysis.csv,
data/sim_robust_mincov.csv, data/emp_cells.csv, data/raw/box*.jsonl,
paper/litreview/claims/claims.csv and data/per_cluster_*.csv (Opus labels).
Writes paper/numbers_more.tex. Run after sim_robust.py and empirical_extra.py.
Run: python3 analysis/extra_macros.py   (standard library only)
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "data", name))))


m = {}

# Unequal sizes: Kish with the plain mean size, skewed design, rho = 0.05.
u = next(r for r in rows("sim_extra_unequal.csv")
         if r["design"].startswith("25") and float(r["rho"]) == 0.05)
m["UneqKishMeanCov"] = f"{float(u['cov_kish_equal']) * 100:.1f}"

# Budget: how much the bound rises when the classical m* is used instead of the optimum,
# over the designs with a positive scenario-writing cost.
rises = [float(r["ub_bb_at_classical"]) / float(r["opt_ub_bb"]) - 1
         for r in rows("sim_extra_budget.csv") if float(r["c_ratio"]) > 0]
m["ClassRiseMin"] = f"{min(rises) * 100:.0f}"
m["ClassRiseMax"] = f"{max(rises) * 100:.0f}"

# Reanalysis rows quoted in the prose.
re_ = {r["id"]: r for r in rows("reanalysis.csv")}
m["ReNaiveFewK"] = f"{float(re_['C6']['naive']) * 100:.1f}"          # 300 runs, K = 5 (same N as C8)
m["ReEmailRhoTwenty"] = f"{float(re_['C6']['bb0.2']) * 100:.1f}"     # K = 5
m["ReCodexRhoTwenty"] = f"{float(re_['C8']['bb0.2']) * 100:.1f}"     # K = 3
m["ReAisiAll"] = f"{float(re_['C10']['naive']) * 100:.1f}"           # K = N
assert abs(float(re_["C10"]["naive"]) - float(re_["C10"]["scen"])) < 1e-12

# Selection funnel for the reanalysis (counts as words for the prose).
WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen".split()
claims = list(csv.DictReader(open(os.path.join(ROOT, "paper", "litreview", "claims", "claims.csv"))))
eligible = [c for c in claims if c["x"] == "0" and c["N"] and c["K"]]
m["ReClaimsFound"] = str(len(claims))
m["ReSources"] = WORDS[len({c["url"] for c in claims})]
m["ReClaimsEligible"] = WORDS[len(eligible)]
m["ReClaimsEligibleCap"] = WORDS[len(eligible)].capitalize()
m["ReClaimsDropped"] = WORDS[len(eligible) - len(re_)]
m["ReClaimsKept"] = WORDS[len(re_)]
m["ReClaimsSingle"] = WORDS[sum(int(r["K"]) == 1 for r in re_.values())]

# Worst-case zero-event bound (two-point law, Proposition 3 of the paper) for the
# zero-blackmail setting, 15 configurations x 100 rollouts, over every ICC measured
# on the suite (both estimators, both label sets).
def wc_zero_upper(rho, K=15, m=100, alpha=0.05):
    import math
    def log_p0(mu):
        c = mu + rho * (1 - mu)
        w = mu / c
        return K * math.log(1 - w + w * (1 - c) ** m)
    lo, hi = 1e-12, 0.999
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if log_p0(mid) >= math.log(alpha) else (lo, mid)
    return lo


cells = rows("emp_cells.csv")
# Largest gap between the two ICC estimators within a behaviour, and the design
# effect at the largest upper limit for residual clustering (Opus labels, m = 100).
import math
m["EmpEstGapMax"] = f"{math.ceil(max(abs(float(r['anova']) - float(r['bb'])) for r in cells) * 100) / 100:.2f}"
m["EmpResidDeffHiMax"] = f"{1 + 99 * max(float(r['obs_res_hi']) for r in cells if r['labels'] == 'Opus'):.1f}"

rhos = [float(r[k]) for r in rows("emp_cells.csv") for k in ("anova", "bb")
        if r[k] not in ("", "nan") and float(r[k]) > 0]
m["EmpZeroWcMin"] = f"{wc_zero_upper(min(rhos)) * 100:.2f}"
m["EmpZeroWcMax"] = f"{wc_zero_upper(max(rhos)) * 100:.2f}"

# Blackmail rollouts that failed the suite's code check but that the gpt-oss-120b
# judge called blackmail (the judge's raw verdict is stored for every rollout).
import glob
import json
miss = {}
for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "box*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r["condition"].startswith("blackmail") and not r["gate"]:
            miss.setdefault(r["job"], [0, 0])
            miss[r["job"]][0] += 1
            miss[r["job"]][1] += bool(r.get("judge_verdict"))
print("gate failures, judge yes:", miss)
zero_jobs = [j for j in miss if j in ("qwen3_8b", "gpt_oss_20b") or "think" in j]
assert len(zero_jobs) == 3, zero_jobs
m["GateMissJudgeYesZeroModels"] = WORDS[sum(miss[j][1] for j in zero_jobs)]
m["GateMissZeroModelsMin"] = f"{min(miss[j][0] for j in zero_jobs):,}".replace(",", "{,}")
m["GateMissJudgeYesMistral"] = str(next(v[1] for j, v in miss.items() if "mistral" in j))

# Rounded worst coverage of the beta-binomial bound under a two-point truth (grid of 200 rates),
# for the abstract and introduction.
mc = [r for r in rows("sim_robust_mincov.csv") if r["truth"] == "twopoint"]
m["RobMinCovTwoPointExactRound"] = f"{min(float(r['exact_true_min']) for r in mc) * 100:.0f}"

# Spread of leaking rates across configurations of one model (Opus labels).
leak = [float(r["rate"]) for r in rows("per_cluster_qwen3_32b.csv") if r["cluster"].startswith("leaking")]
m["LeakSpreadMin"] = f"{min(leak) * 100:.0f}"
m["LeakSpreadMax"] = f"{max(leak) * 100:.0f}"

with open(os.path.join(ROOT, "paper", "numbers_more.tex"), "w") as fh:
    fh.write("% generated by analysis/extra_macros.py; do not edit\n")
    for k, v in m.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(m)
