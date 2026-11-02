#!/usr/bin/env python3
"""Flatten agentic-misalignment results into the CSV read by analysis/analyze_real.py.

Reads   <results_dir>/models/<model_dir>/<condition>/sample_NNN/response.json
Writes  model,cluster,scenario,goal_type,goal_value,urgency,variant,seed,harmful

Conditions are named by generate_prompts.create_condition_id:
    {scenario}_{goal_type}-{goal_value}_{urgency_type}[_prod]
A prompt-wording variant is a copied condition directory with the suffix
"__<variant>" (see RUNBOOK.md); without the suffix the variant is "base".
`cluster` is the condition without the variant suffix, `seed` is the suite's
sample_id (a replicate index; the suite sets no RNG seed), and `harmful` is
classification.harmful_behavior as 0/1.

Files with no usable label (failed calls, unclassified, classifier error) are
skipped and counted on stderr. Stdlib only.

Usage: python3 to_csv.py results/qwen3_8b_clusters -o data/rollouts.csv [--model-label qwen3-8b]
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

COLUMNS = ["model", "cluster", "scenario", "goal_type", "goal_value", "urgency", "variant", "seed", "harmful"]


def parse_condition(name):
    base, _, variant = name.partition("__")
    parts = base.split("_")
    if parts[-1] == "prod":
        parts = parts[:-1]
    if len(parts) < 3 or "-" not in parts[1]:
        raise ValueError(f"unrecognised condition name: {name}")
    goal_type, goal_value = "_".join(parts[1:-1]).split("-", 1)
    return {"cluster": base, "scenario": parts[0], "goal_type": goal_type,
            "goal_value": goal_value, "urgency": parts[-1], "variant": variant or "base"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path, help="results/<experiment_id>")
    ap.add_argument("-o", "--output", type=Path, help="CSV path (default: stdout)")
    ap.add_argument("--model-label", help="value for the model column (default: metadata.model)")
    args = ap.parse_args()

    files = sorted((args.results_dir / "models").glob("*/*/sample_*/response.json"))
    if not files:
        sys.exit(f"no response.json under {args.results_dir / 'models'}")

    rows, skipped = [], Counter()
    for f in files:
        condition, model_dir = f.parent.parent.name, f.parent.parent.parent.name
        data = json.loads(f.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        label = data.get("classification", {}).get("harmful_behavior")
        if "error" in data or "raw_response" not in data:
            skipped["failed call (error file)"] += 1
            continue
        if not isinstance(label, bool):
            skipped["no classification.harmful_behavior"] += 1
            continue
        row = parse_condition(condition)
        row["model"] = args.model_label or meta.get("model", model_dir)
        row["seed"] = int(meta.get("sample_id", f.parent.name.split("_")[1]))
        row["harmful"] = int(label)
        rows.append(row)

    rows.sort(key=lambda r: (r["model"], r["cluster"], r["variant"], r["seed"]))
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    with out:
        w = csv.DictWriter(out, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    per_cluster = Counter((r["model"], r["cluster"]) for r in rows)
    events = Counter((r["model"], r["cluster"]) for r in rows if r["harmful"])
    print(f"{len(files)} response files, {len(rows)} rows written, "
          f"{len(per_cluster)} clusters, {sum(events.values())} harmful", file=sys.stderr)
    if per_cluster:
        print(f"rows per cluster: min {min(per_cluster.values())}, max {max(per_cluster.values())}", file=sys.stderr)
    for reason, n in skipped.items():
        print(f"skipped {n}: {reason}", file=sys.stderr)
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
