# Clustered rollouts (G022)

Zero-event safety bounds that assume independent rollouts are too narrow when the rollouts come from a few dozen scenarios.

## Layout

- `analysis/clusterstats.py`: ICC estimators (ANOVA and beta-binomial), design effect, Clopper-Pearson bounds, cluster bootstrap.
- `analysis/sim.py`: figures, coverage simulation, `paper/numbers_sim.tex`, `paper/tab_coverage.tex`. Run with `uv run analysis/sim.py` (about 40 s on CPU).
- `analysis/analyze_real.py data/rollouts.csv`: per-model ICC, DEFF and bounds from real rollouts. Writes `paper/numbers_real.tex`. CSV columns: `model,cluster,scenario,goal_type,goal_value,urgency,variant,seed,harmful`.
- `paper/main.tex`: the draft. Build with `tectonic main.tex` from `paper/`.
- `paper/litreview/NOVELTY_NOTE.md`: the Step 0 novelty check.

## Status (2026-10-22)

- Done: novelty check, simulation, figures, analysis code, a 4-page draft with [TODO] markers.
- Not run: the GPU experiment on the agentic misalignment suite. No real-rollout numbers exist yet.
