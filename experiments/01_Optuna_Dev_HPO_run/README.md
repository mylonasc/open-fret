# 01 — Optuna dev-HPO run

Status: **running** (study `dev-hpo-v4`, 12 trials, 5 epochs each, 45-min backstop).

- Data: `dev-hpo-split-v1` — group-disjoint 5% dev split (68,794 train / 8,864 val).
- Search: lr, dropout, burn-in `warmup_examples` {10k,50k,100k}, LR step-decay
  at `decay_after_examples` {10k,50k,100k} by `decay_factor` {0.1,0.3,0.5},
  batch {32…256} × grad-accum {1,2} (VRAM saturation arm).
- Per-trial telemetry: train/val loss curves, GPU util + memory curves,
  peak VRAM, stability flag (`trial_<n>_history.json` + Optuna user attrs).
- `losses.ipynb`: train vs validation loss plots + summary table.
- `batch_saturation.csv` (in `models/hpo-dev-v1/`, written at study end):
  batch vs memory vs stability.
