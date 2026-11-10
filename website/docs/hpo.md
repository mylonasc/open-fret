---
sidebar_position: 4
---

# Hyper-parameter optimization

Two Optuna programs ran in this project: a broad **dev HPO** over the
training schedule, and **loss-ablation studies** with fixed weights. Both
are reproducible from the repo: `scripts/hpo_optuna.py` (search) and
`scripts/train_model.py` (single runs). Study databases (`hpo-dev.db`,
`hpo-losses.db`) record every trial; resuming with changed loss flags or
tokenizer metadata is rejected, so each ablation gets its own study.

## Dev HPO: what actually mattered

Study `dev-hpo-v4` (12 trials, minimize validation loss) searched learning
rate, dropout, batch size, gradient accumulation, warmup length and the
burn-in/decay schedule. Best trial: **eval loss 2.502** at

| hyper-parameter | best value |
|---|---|
| learning rate | $5.7\times10^{-4}$ |
| dropout | 0.144 |
| batch size | 128 (× grad-accum 2 → effective 256) |
| warmup examples | 10,000 |
| decay after examples | 100,000 (×0.1) |

Trial values spanned **2.50–7.41**, and the spread is almost entirely
explained by two knobs — parameter importances from the finished study:

| hyper-parameter | importance |
|---|---|
| learning rate | 0.60 |
| dropout | 0.39 |
| batch size / grad-accum / schedule | ≈ 0.01 combined |

The lesson transferred directly into the full runs: the curated 3-epoch run
uses lr $5.7\times10^{-4}$, dropout 0.144, batch 128 × 2 accumulation, the
same warmup/decay — i.e. the dev-HPO winner became the full-run schedule
(`hyperparams` in `models/full-curated-3ep-20261109/run.json`).

Reading the trial table, the pattern is stark: every trial with
lr $\lesssim 3\times10^{-5}$ sits above 4.1, while the three trials near or
above $10^{-4}$ take the top three spots. Dropout modulates within that —
the winner's 0.144 beats both near-zero (0.0017 → 6.72) and high (0.17 →
2.70) extremes at comparable learning rates.

## Loss-ablation studies

Loss weights are **fixed per study** (deliberately not part of the search
space), each with its own study name and output root:

| study | objective | best |
|---|---|---|
| `full-combined-ce03-pitch1-smooth01-v1` | $0.3\cdot\mathrm{CE} + 1.0\cdot\mathrm{pitch} + 0.1$ smoothing | 2.562 (1 trial) |
| `full-curated-ce03-pitch1-smooth01-type03-v2` | above $+ 0.3\cdot\mathrm{type}$ + conditional pitch | running (2 trials) |

These studies exist to produce comparable checkpoints, not to declare a
winner: per the [losses roadmap](losses.md#roadmap), each term still needs a
same-decoder CE-control comparison on *generated-note* metrics before any
promotion claim. Scalar `eval_loss` values across different objectives are
not comparable — the study metadata records the full loss config alongside
every trial so nobody mixes them up. To make the comparison audible later,
pass the sample panel explicitly — every completed trial then writes playable
fixed-window examples connected to its config:

```bash
python scripts/hpo_optuna.py --study-name span-v1 ... \
  --sample-data-dir dataset/processed/loss-v4-curated-20261109/individual \
  --sample-val-files dataset/processed/loss-v4-curated-20261109/val_files.txt \
  --sample-songs 5 --span-alpha 0.5
```

This yields `trial_<n>/samples/` (`.input/.pred/.ref.txt`, `.tex`,
`manifest.json`) plus `viewer.html`, with the song count, viewer path and
validation loss recorded in the trial user attrs (hence in
`study_summary.json`).

## Practical notes

- HPO never touches held-out test data; selection is validation-only.
- `batch_saturation.csv` (in `models/hpo-dev-v1/`) records GPU
  utilization vs. batch size for capacity planning.
- The winning schedule trains the 843k-window curated corpus at ~22.9
  eval steps/s on an RTX 4090 (fp16), with checkpoints every 2,454 steps.
