# Fresh combined-loss full run

Run: `models/full-combined-loss-3ep-20261108/` (initially running).

## Objective

`total = 0.3 * (CE + smoothing_delta(epsilon=0.1)) + 1.0 * pitch_marginal`

CLI: `--ce-loss-weight 0.3 --pitch-loss-alpha 1 --equivalence-smoothing 0.1`.
The smoothing adjustment is part of CE and receives the same 0.3 weight.
The default CE-only configuration is unchanged. Combining these terms was
explicitly requested; their musical-quality benefit has not yet been measured.

## Rebuilt data and supported representation

- New root: `dataset/processed/loss-v3-alphatab-combined-20261108/`.
- Pinned Node dependency: AlphaTab 1.3.0, `npm ci --prefix tools/alphatex_parser`.
  It is an external MPL-2.0 dependency, not vendored code.
- Import normalization translates corpus octave names to scientific pitch,
  quotes unnamed track headers, and removes non-musical `\frets` metadata that
  this importer version does not accept. AlphaTab handles the actual grammar.
- Note timing uses AlphaTab's MIDI duration routine: inline inheritance,
  dots/tuplets, tied-note duration/no rearticulation, palm mute, staccato and
  let-ring are exercised by regression tests.
- Targets are explicitly a **base-fretted-pitch/fingering projection**.
  Bend/slide/harmonic/articulation nuances are counted in the processing
  manifest, not claimed as modeled expressive targets. Repeat expansion,
  variable-tempo exports, over-range frets and incompatible notation are rejected.
- 47,299 files audited; 13,035 rejected; **1,054,200 windows** produced.
  Articulation projection counts and failure reasons are retained in
  `individual/_processing_manifest.json` and `individual/_failures.log`.
- Cache: **843,886 train / 105,394 validation / 104,920 test**, zero cache
  skips/truncations. The held-out test is cached but not evaluated for selection.
- 6,223 vocabulary entries: training-derived tokens plus a predefined
  fretboard, MIDI pitches, capo and short-tick vocabulary. No held-out target
  scanning for vocabulary construction. Source tuning/capo reach the loss
  separately, including unseen composite tuning prefixes.
- Splits are source-group disjoint. Existing limitations around train-only
  augmentation and cross-file duplicate auditing remain in #5. This corrected,
  smaller corpus is **not** the same-dataset benchmark as the previous run.

## Training and inspection

Fresh random T5 initialization: no `--resume` or `--init-from`. Three epochs,
seed 42, same tiny architecture, batch 128 × accumulation 2, LR
0.0005696488989999719, dropout 0.14446443138833517, warmup 39 steps and decay
at step 391 by 0.1, matching the previous optimizer protocol.

Validation every **1,227 steps**, checkpoints every **2,454 steps**, scaling the
previous cadence by epoch size. Every validation/checkpoint exports five fixed
validation songs and renders HTML automatically:

- `<run>/samples/index.html`: refreshing history index.
- `<run>/samples/step-<N>/viewer.html`: playable predictions/input MIDI and
  expandable validation/component metrics; source tuning/capo/tempo respected.
- `<run>/checkpoint-<N>/viewer.html`: link to that checkpoint's samples.

Internet access is needed for viewer CDN scripts/soundfont. Exported predictions
can still contain model errors; rendering/listening is not a quality guarantee.

Optuna: `sqlite:///hpo-losses.db`, study
`full-combined-ce03-pitch1-smooth01-v1`. This full run is a fixed-configuration
trial, not another hyperparameter search. `train_component_curve` and
`eval_component_curve` record total loss, CE, pitch marginal, smoothing delta,
weighted components, weights and eligible counts. They are also written to
`<run>/loss_components.json`, including before pruning/failure in HPO studies.
Component averages have token/note-specific denominators; they need not sum
exactly to Trainer's differently batch-weighted aggregate `eval_loss`.

`run.json` records model initialization, loss weights, tokenizer hash, data,
schedule and study/trial. `full_run_plan.json` in the data root records the exact
launch command. Training log: `<run>/train.log`; full console log:
`<data-root>/train.log`. Old runs/data/cache were left untouched.

Orchestration is `scripts/run_full_loss_experiment.py` with `--prepare-only`
and `--train-only` modes. It requires completed extended-parser processing and
refuses a nonempty run directory. Do not relaunch the currently active run.

Synthetic GPU preflight (RTX 4090, actual rebuilt vocabulary, batch 128 × 512):
combined forward/backward ~0.151 s and peak allocation ~10.95 GB; finite
gradients. This is not full-pipeline throughput or a quality metric.
