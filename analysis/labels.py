"""Labels for the real rollouts, and the merge into rollouts.csv.

  python3 analysis/labels.py sample [n]   # blind Opus batches: stratified sample of gate-passing rollouts
  python3 analysis/labels.py merge        # data/raw/*.jsonl (+ data/labels/*.jsonl) -> data/rollouts.csv

data/raw/*.jsonl comes from the box-side export: one JSON row per rollout with
the suite's judge-independent check ("gate"), the local judge's label ("judge",
which is gate AND judge verdict, as the suite computes it) and, when the gate
passes, the text with its forwarded-email context.

The primary label is the local judge (gpt-oss-120b through the suite's own
classify_results.py). Opus subagents label a blind, stratified sample; merge
reports their agreement with the judge. Batches carry only an opaque id, the
scenario type and the text, never the model or condition.
"""
import csv
import glob
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, BATCH, LAB = (os.path.join(ROOT, "data", d) for d in ("raw", "label_batches", "labels"))


def rid(r):
    return hashlib.sha1(f"{r['job']}|{r['condition']}|{r['sample']}".encode()).hexdigest()[:12]


def load_raw():
    rows = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line.startswith("{"):
                r = json.loads(line)
                rows[rid(r)] = r  # a later export of the same rollout replaces an earlier one
    return rows


def sample(n=300, batch_size=50, jobs=None, prefix="batch"):
    """Stratify by job x scenario x judge label so both yes and no are checked.
    `jobs` (comma-separated) limits the sample to those jobs; `prefix` names the batch files."""
    rows = load_raw()
    keep = set(jobs.split(",")) if jobs else None
    strata = defaultdict(list)
    for k, r in rows.items():
        if r["gate"] and "judge" in r and (keep is None or r["job"] in keep):
            strata[(r["job"], r["condition"].split("_")[0], r["judge"])].append(k)
    rng = random.Random(20261022)
    per = max(1, n // max(1, len(strata)))
    chosen = []
    for key in sorted(strata):
        ks = sorted(strata[key])
        rng.shuffle(ks)
        chosen += ks[:per]
    rng.shuffle(chosen)
    os.makedirs(BATCH, exist_ok=True)
    for f in glob.glob(os.path.join(BATCH, f"{prefix}_*.jsonl")):
        os.remove(f)
    for i in range(0, len(chosen), batch_size):
        with open(os.path.join(BATCH, f"{prefix}_{i // batch_size:02d}.jsonl"), "w") as fh:
            for k in chosen[i:i + batch_size]:
                r = rows[k]
                fh.write(json.dumps({"id": k, "scenario": r["condition"].split("_")[0],
                                     "text": r["text"], "email_context": r.get("email_context", "")}) + "\n")
    print(f"{len(strata)} strata, {per} per stratum, {len(chosen)} rollouts in "
          f"{-(-len(chosen) // batch_size)} batches")


def full(batch_size=100, prefix="c"):
    """Blind batches of every gate-passing rollout that has no Opus label yet."""
    rows = load_raw()
    done = set()
    for f in glob.glob(os.path.join(LAB, "*.jsonl")):
        done |= {json.loads(l)["id"] for l in open(f) if l.strip()}
    todo = sorted(k for k, r in rows.items() if r["gate"] and k not in done)
    random.Random(20261023).shuffle(todo)
    os.makedirs(BATCH, exist_ok=True)
    for f in glob.glob(os.path.join(BATCH, f"{prefix}_*.jsonl")):
        os.remove(f)
    for i in range(0, len(todo), batch_size):
        with open(os.path.join(BATCH, f"{prefix}_{i // batch_size:02d}.jsonl"), "w") as fh:
            for k in todo[i:i + batch_size]:
                r = rows[k]
                fh.write(json.dumps({"id": k, "scenario": r["condition"].split("_")[0],
                                     "text": r["text"], "email_context": r.get("email_context", "")}) + "\n")
    print(f"{sum(r['gate'] for r in rows.values())} gate-passing, {len(done)} already labelled, "
          f"{len(todo)} to label in {-(-len(todo) // batch_size)} batches of <= {batch_size}")


def merge():
    rows = load_raw()
    opus = {}
    for f in glob.glob(os.path.join(LAB, "*.jsonl")):
        for line in open(f):
            if line.strip():
                x = json.loads(line)
                opus[x["id"]] = x["answer"].strip().lower() == "yes"
    unjudged = [k for k, r in rows.items() if "judge" not in r]
    # Strict Opus labels (overrides from ADJUDICATION.md applied) for every gate-passing rollout.
    strict_all = dict(opus)
    ov_path = os.path.join(LAB, "strict_overrides.jsonl.txt")
    if os.path.exists(ov_path):
        for line in open(ov_path):
            if line.strip():
                x = json.loads(line)
                strict_all[x["id"]] = x["answer"].strip().lower() == "yes"
    unlabelled = [k for k, r in rows.items() if r["gate"] and k not in strict_all]
    order = sorted(rows.items(), key=lambda kv: (kv[1]["job"], kv[1]["condition"], kv[1]["sample"]))
    header = ["model", "cluster", "scenario", "goal_type", "goal_value", "urgency", "variant", "seed", "harmful"]
    outs = {"rollouts_judge.csv": lambda k, r: int(r["judge"])}
    if not unlabelled:  # Opus labels are primary once every gate-passing rollout has one
        outs["rollouts_opus.csv"] = lambda k, r: int(bool(r["gate"] and strict_all.get(k, False)))
    for name, lab in outs.items():
        with open(os.path.join(ROOT, "data", name), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for k, r in order:
                if "judge" not in r:
                    continue
                scen, goal, urg = r["condition"].split("_")
                gtype, gval = goal.split("-", 1)
                w.writerow([r["job"], r["condition"], scen, gtype, gval, urg, gval, r["sample"], lab(k, r)])
    print(f"gate-passing rollouts without an Opus label: {len(unlabelled)}; wrote {sorted(outs)}")
    # Agreement of the judge with Opus on the labelled sample (both conditional on the gate).
    # "strict" applies data/labels/strict_overrides.jsonl.txt (see ADJUDICATION.md).
    strict = dict(opus)
    ov = os.path.join(LAB, "strict_overrides.jsonl.txt")
    if os.path.exists(ov):
        for line in open(ov):
            if line.strip():
                x = json.loads(line)
                strict[x["id"]] = x["answer"].strip().lower() == "yes"

    def agree(lab):
        pairs = [(rows[k]["judge"], v) for k, v in lab.items() if k in rows and "judge" in rows[k]]
        tp = sum(j and o for j, o in pairs); tn = sum(not j and not o for j, o in pairs)
        fp = sum(j and not o for j, o in pairs); fn = sum(not j and o for j, o in pairs)
        n = len(pairs)
        po = (tp + tn) / n if n else float("nan")
        pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / n**2 if n else float("nan")
        kappa = (po - pe) / (1 - pe) if n and pe < 1 else float("nan")
        return dict(n=n, tp=tp, tn=tn, fp=fp, fn=fn, agreement=po, kappa=kappa)

    summary = dict(rollouts=len(rows), unjudged=len(unjudged), strict=agree(strict), raw=agree(opus))
    gate = {j: [0, 0] for j in sorted({r["job"] for r in rows.values()})}
    for r in rows.values():
        gate[r["job"]][0] += r["gate"]; gate[r["job"]][1] += 1
    m = {"JudgeN": summary["strict"]["n"], "TotalRollouts": f"{len(rows):,}".replace(",", "{,}"),
         "GatePass": f"{sum(g[0] for g in gate.values()):,}".replace(",", "{,}")}
    for tag, s in (("Strict", summary["strict"]), ("Raw", summary["raw"])):
        m[f"JudgeAgree{tag}"] = f"{s['agreement'] * 100:.1f}"
        m[f"JudgeKappa{tag}"] = f"{s['kappa']:.2f}"
        m[f"JudgeFP{tag}"] = s["fp"]
        m[f"JudgeFN{tag}"] = s["fn"]
        m[f"JudgeYes{tag}"] = s["tp"] + s["fp"]
    with open(os.path.join(ROOT, "paper", "numbers_judge.tex"), "w") as fh:
        fh.write("% generated by analysis/labels.py merge; do not edit\n")
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    json.dump(summary, open(os.path.join(ROOT, "data", "judge_agreement.json"), "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    {"sample": lambda: sample(int(sys.argv[2]), int(sys.argv[3]), *sys.argv[4:6]),
     "full": lambda: full(*(int(a) for a in sys.argv[2:3])),
     "merge": merge}[sys.argv[1]]()
