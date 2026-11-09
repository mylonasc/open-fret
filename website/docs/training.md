---
sidebar_position: 5
---

# Training

## Setup shared by all runs

- **Architecture.** A small T5 encoder-decoder (`d_model` 128, `d_ff`
  1024, 3 encoder + 3 decoder layers, 4 heads) with a task WordLevel
  tokenizer — configured in `scripts/config.py`, trained **from scratch**
  (`init_from: null`). `--resume` and `--init-from` are mutually exclusive:
  resuming continues an Optuna trial, initializing starts a new fine-tune.
- **Hardware/precision.** Single NVIDIA RTX 4090, fp16, seed 42.
- **Schedule.** The dev-HPO winner (see [HPO](hpo.md)): lr
  $5.7\times10^{-4}$, dropout 0.144, batch 128 × grad-accum 2 (effective
  256), 10k-example warmup, ×0.1 decay after 100k examples.
- **Supervision.** One fingering slot per input attack (anchored decoding at
  sample time: beams 4, pitch-equivalent masking, distinct strings per
  onset); unsupported input is reported, never hallucinated.
- **Selection.** Validation metrics only. A frozen five-window panel is
  decoded every validation round with playable viewers (`samples/step-*/`,
  `viewer.html` + `index.html` history); the held-out test split is cached
  but never used for tuning.

## Runs

| Run | Corpus (windows) | Vocab | Objective | Epochs | Status |
|---|---|---|---|---|---|
| `full-hpo-best-3ep-20261108-1530` | 1,375,102 (legacy rebuilt) | 4,402 | plain CE | 3 | complete (5 ckpts) |
| `full-combined-loss-3ep-20261108` | 843,886 (loss-v3) | 6,223 | $0.3\,\mathrm{CE} + 1.0\,\mathrm{pitch} + 0.1$ smooth | 3 | complete (5 ckpts, to step 9891) |
| `full-curated-3ep-20261109` | 843,886 (loss-v4 curated) | 6,631 | above $+ 0.3\,\mathrm{type}$ + conditional pitch | 20 | running (5 ckpts, 42 sample rounds) |
| `tiny-tab-b96x2` / `tiny-tab-v1*` | 1,375,102 / fixtures | 4,402 | plain CE | 3 | smoke / fixture runs |

Corpus notes:

- **loss-v3** (`loss-v3-alphatab-combined-20261108`): first AlphaTab-1.3.0
  rebuild with MIDI-duration timing; 388 `[UNK]` validation labels remained.
- **loss-v4** (`loss-v4-curated-20261109`): same parser/splits/schedule,
  plus structural conditioning lists (vocab 6,223 → 6,631, **zero `[UNK]`**),
  group-level curation masking (441k windows, 23.6M note tokens), the
  event-type term and conditional pitch. Splits are byte-identical to v3, so
  anchored-generation metrics compare across the two systems on the same
  windows — confounded by curation+vocab+loss changes, i.e. not an isolated
  ablation.

Every run directory records `run.json` (loss config, token-metadata hash,
hyper-parameters, Optuna study/trial), `train.log` (per-step component
losses: `ce_loss`, `pitch_loss`, `equivalence_smoothing_delta`, …),
`loss_components.json`, checkpoints, and per-step `samples/` with
`.input.txt` / `.pred.txt` / `.ref.txt` triples plus viewers. The current
production checkpoint is the curated run's latest; weights are not published
(training-data rights unresolved — see [Dataset](dataset.md)).

## What the logs show (curated run, late training)

```
Step: 051950 | Train Loss: 2.2377 | LR: 5.70e-05 | ce_loss: 1.92727
  | pitch_loss: 1.45400 | equivalence_smoothing_delta: 0.12958
```

Component losses are teacher-forced diagnostics, not quality measures —
compare runs on the frozen panel's generated-note metrics instead.
