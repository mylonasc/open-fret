# Experimental pitch-aware training losses

These are opt-in ablations, not demonstrated model-quality improvements.
See [the combined full-run protocol and artifacts](full_combined_run.md) for the
active user-requested experiment, extended parser, and current loss weights.
Without either flag, training delegates to Hugging Face's original CE loss.
The model architecture, tokenizer format and inference interface are unchanged.

## Fix the data first

The original string-as-duration bug in #2 was already fixed. A further bug was
confirmed against AlphaTab 1.3.0: inline durations persist. For example,
`:4 0.1.8 0.2` means **two eighth notes**, not an eighth then a quarter.
The strict parser now preserves this behavior for notes, chords and rests, handles
dotted beats, and rejects malformed/unsupported syntax instead of mining valid
substrings or stripping effect blocks. Track titles are excluded from beat syntax.

Supported subset: `:D`, `F.S[.D]`, chords with an outer duration, rests, bar
separators, `{d}`/`{dd}` beat dots, and legacy dotted/suffix b/s/v syntax. Ties,
tuplets, repeat multipliers, tempo changes, other effect blocks, multi-staff
directives and durations outside 1/2/4/8/16/32/64 fail visibly. A rejected file
produces no tracks and is counted/logged by the processing entry point. The
parser is deliberately a strict subset, not a full AlphaTex importer.

The default processing backend is now AlphaTab 1.3.0 (install with
`npm ci --prefix tools/alphatex_parser`), extending support to common effect
syntax, tuplets and ties. It uses the importer's own MIDI note durations and
records unmodeled articulation features explicitly. Use `--parser-backend strict`
for the small dependency-free grammar described above. Extended windows use
`alphatab-1.3.0-midi-duration-v1`; strict windows use
`alphatex-v2-strict-duration-inheritance`. Caches require a consistent supported
version. **Existing production artifacts were preserved, not overwritten.**
New domain-aware training refuses old/unversioned caches; the old CE
path can still load legacy caches for reproducibility. Do not rename a manifest
version to bypass the check: its timing supervision is actually different.

Rebuild into new paths; never mix old and new individual-window directories:

```bash
python scripts/01_process.py --extracted-dir dataset/raw/extracted \
  --processed-dir dataset/processed/loss-v2/individual
python scripts/03_split.py --input-dir dataset/processed/loss-v2/individual \
  --output-dir dataset/processed/loss-v2
python scripts/01.5_generate_vocab.py --input-dir dataset/processed/loss-v2/individual \
  --train-list dataset/processed/loss-v2/train_files.txt \
  --output dataset/processed/loss-v2/unique_tokens.json \
  --structural-lists dataset/processed/loss-v2/train_files.txt,dataset/processed/loss-v2/val_files.txt,dataset/processed/loss-v2/test_files.txt
python scripts/02_tokenize.py --vocab dataset/processed/loss-v2/unique_tokens.json \
  --output-dir dataset/processed/loss-v2/tab_tokenizer
python scripts/04_cache_dataset.py --data-dir dataset/processed/loss-v2/individual \
  --tokenizer-dir dataset/processed/loss-v2/tab_tokenizer \
  --train-list dataset/processed/loss-v2/train_files.txt \
  --val-list dataset/processed/loss-v2/val_files.txt \
  --test-list dataset/processed/loss-v2/test_files.txt \
  --cache-dir dataset/processed/loss-v2/cache
```

`--structural-lists` adds `CAPO_*`, `TUNING_*`, `NO_*`, `NF_*` and `TS_*` tokens
from the listed splits. These are input conditioning/timing structure, not
target fingerings, so this is not target leakage; `TAB_*` targets remain
train-only. Without it, validation tuning prefixes and rare large time shifts
were silently encoded as `[UNK]`. The cache builder now rejects any remaining
out-of-vocabulary token instead of teaching on a substituted label.

### Same-onset/same-string curation

Some reference windows assign two different frets to one string inside a
single attack group (overlapping voices or malformed source bars). Keeping them
teaches the model an impossible chord. `open_fret.curation` identifies each
affected group and the cache masks the **whole group** (its `TAB_*` labels
become `-100`) while retaining the time shift and every unaffected onset.
`cache_manifest.json` records per-split and total `scanned_windows`,
`masked_windows` and `masked_notes`; use `--no-mask-conflicts` only to reproduce
legacy caches for comparison. Masking is by onset, not by window, because a
minority of onsets is affected even though many windows contain one.


These commands rebuild timing; they do **not** resolve #5's training-only
augmentation/duplicate audit. Review `_failures.log` for rejected syntax before
accepting the new corpus. Use the same rebuilt data/tokenizer for CE controls
and both loss experiments. A changed vocabulary must not be resumed against an
old checkpoint by token-ID coincidence. Prefer fresh matched-seed experiments.

## Two separate experiments

The following examples use the rebuilt paths above:

```bash
# CE + pitch marginal, initially alpha=0.3
python scripts/train_model.py --run-name pitch-v1 \
  --cache-dir dataset/processed/loss-v2/cache \
  --tokenizer-dir dataset/processed/loss-v2/tab_tokenizer \
  --data-dir dataset/processed/loss-v2/individual \
  --val-files dataset/processed/loss-v2/val_files.txt \
  --pitch-loss-alpha 0.3

# Alternative: equivalence-aware smoothing, epsilon=0.1
python scripts/train_model.py --run-name smoothing-v1 \
  --cache-dir dataset/processed/loss-v2/cache \
  --tokenizer-dir dataset/processed/loss-v2/tab_tokenizer \
  --data-dir dataset/processed/loss-v2/individual \
  --val-files dataset/processed/loss-v2/val_files.txt \
  --equivalence-smoothing 0.1
```

Use the same budget/seed/schedule flags for each comparison and a fresh CE control
without either loss flag. The flags can also be combined explicitly. Zero disables
either feature. `--loss-max-fret` defaults to 24; targets outside configured
string/fret limits fail rather than silently becoming incorrect pitch examples.

### Pitch marginal

For a TAB target at a teacher-forced position, find vocabulary tokens with the
same base fretted MIDI pitch under that example's tuning/capo:

`L = CE + alpha * mean_note_positions(-log(sum_same_pitch p(token)))`.

The sum uses the full vocabulary softmax, so predicting TS/EOS is not a free
pitch match. Technique variants may share base pitch in this auxiliary term;
ordinary CE still supervises the exact technique/fingering. This does not model
the continuous pitch contour of a bend. It does not align free-running outputs.

### Conditional pitch marginal

`--conditional-pitch` subtracts the playable-TAB normalizer from the pitch
marginal: it maximizes `P(correct pitch | any playable TAB token)` instead of
`P(correct pitch)` over the full vocabulary. Without it, the auxiliary term also
pressures the model to emit more TAB tokens, which distorts the TAB/TS/EOS
balance and can inflate note counts. Exact CE still decides when notes occur.
This option requires `--pitch-loss-alpha > 0` to have an effect.

### Event-type marginal

`--event-type-alpha 0.3` adds a macro-averaged marginal loss over three observed
classes: melodic TAB note, time shift, and EOS. Each class is normalized by its
own count before averaging, so EOS is not drowned by the much more frequent note
and time tokens. Other structural/technique tokens retain exact CE supervision.

### Equivalence smoothing
At technique-free TAB targets with alternatives, keep probability `1-epsilon`
on the reference and split `epsilon` uniformly across other **plain same-pitch**
tokens present in the vocabulary. All other positions retain hard CE, including
technique targets, TS/EOS, and notes with no vocabulary alternative.
The resulting soft CE is averaged over all non-padding target positions.
Padding (`-100`) contributes nothing. No-note/all-ignored custom-loss batches
have finite auxiliary loss zero. All-ignored batches have total loss zero.

### Positional distance, span hinge, neighbourhood stretch

Three opt-in playability terms share the candidate tables (no vocab² anywhere)
and reuse expected frets — softmax over each string's candidate TAB tokens,
never argmax. See the website [losses page](../../website/docs/losses.md) for
the math; the pattern mirrors the pitch marginal.

```bash
python scripts/train_model.py --run-name play-v1 \
  --cache-dir dataset/processed/loss-v4-curated-20261109/cache \
  --tokenizer-dir dataset/processed/loss-v4-curated-20261109/tab_tokenizer \
  --position-alpha 0.5 --span-alpha 0.5 --span-margin 4 \
  --stretch-alpha 0.5 --stretch-margin 5
```

- `--position-alpha`: mean |E[fret] − target| over fretted notes. Prices the
  *where*, not just the pitch (Noise2Fret positional distance formulation).
- `--span-alpha` / `--span-margin` (default 4): hinge on the per-onset
  expected max−min fret (opens excluded) — the midi-to-tab stretch quantity
  and Noise2Fret hand-span hinge in one term. On loss-v4, 98.9% of onsets
  span ≤ 4 frets, so the default margin fires only on the top ~1% shapes.
- `--stretch-alpha` / `--stretch-margin` (default 5): hinge on |E[fret] −
  ±5-note local mean| — the trainable form of midi-to-tab §4.4's
  MAX_DEVIATION relocation rule. Prices jumps; span prices chord width.
- Decoding side (no training): `open_fret.postprocess.relocate_deviating_notes`
  implements §4.4 verbatim (open string preferred, else closest-to-mean free
  string, pitch preserved by construction); opt in with
  `scripts/inference.py --relocate-deviating`.

All three log raw and weighted components (`position_loss`, `span_loss`,
`stretch_loss`, `span_groups`) with the same per-microbatch means convention.
Loss weights are fixed per Optuna study (same-study reuse with changed loss
flags is rejected); tune combinations by comparing separate studies on the
frozen panel, exactly like the existing ablations. Do not compare scalar
`eval_loss` across different objectives.

Equivalence is token-local: alternatives need not be jointly playable in a chord.
There is no timing/occupancy/playability regularizer in this slice. Pitch-aware
supervision can lower exact reference-fingering agreement while preserving pitch.

## Metadata, logging, resume and HPO

- String 1 is the high string. Tuning prefixes contain uncapoed concert MIDI;
  capo is added exactly once. `-1` in a tuning marks an unavailable string,
  never a sounding pitch. No standard-tuning fallback is permitted for custom
  losses; input must start with known `CAPO_*`, then `TUNING_*` tokens.
- Source tuning/capo are also carried separately through the cache and Trainer;
  unseen composite tuning prefixes are not guessed from `[UNK]`. Prefix-only
  callers must still supply valid known conditioning. Known prefixes and source
  metadata must agree.
- The metadata manifest hashes ordered vocabulary, convention and fret limit.
  It and the loss config are recorded in `run.json`, final training metadata,
  model/checkpoint config, and Optuna trial metadata/history/best-config export.
- Logs distinguish `ce_loss`, `pitch_loss`, `equivalence_smoothing_delta`,
  note/eligible smoothing counts and weights; evaluation uses `eval_` prefixes.
  Gradient norms come from Trainer's normal logging. Metrics here are
  teacher-forced losses, not generated-note F1. `eval_loss` is the selected
  objective; do not compare different objectives' scalar values as quality.
- Custom objectives use per-microbatch means; gradient accumulation averages
  those means. Single-device CPU/CUDA paths are tested, not globally weighted
  distributed/multi-device reductions.
- Resume with the identical loss flags. Changed loss or custom-loss metadata
  causes an error before checkpoint restoration. To intentionally change an
  objective, initialize a new run with `--init-from` and an ID-compatible
  tokenizer; this is fine-tuning, not optimizer-state resume.
- Optuna accepts the same flags. Use separate `--study-name` and `--output-root`
  per loss ablation; resuming a study with changed loss or recorded tokenizer
  metadata is rejected. Loss weights are fixed per study in this slice; they
  are not silently added to the hyperparameter search space. Tune them by
  comparing studies, one weight combination per study, on the frozen
  validation panel — and every completed trial now leaves playable generated
  examples connected to its config (`trial_<n>/samples/` + `viewer.html`,
  `songs`/`viewer` recorded in the trial user attrs and thus in
  `study_summary.json`), so weight choices can be judged later by listening,
  not just by `eval_loss`:

```bash
python scripts/hpo_optuna.py --study-name span-v1 --output-root models/hpo-span-v1 \
  --storage sqlite:///hpo-losses.db --cache-dir dataset/processed/loss-v4-curated-20261109/cache \
  --tokenizer-dir dataset/processed/loss-v4-curated-20261109/tab_tokenizer \
  --sample-data-dir dataset/processed/loss-v4-curated-20261109/individual \
  --sample-val-files dataset/processed/loss-v4-curated-20261109/val_files.txt \
  --sample-songs 5 --span-alpha 0.5
```

## Independent loss/label audit

`scripts/audit_losses.py` is read-only: it bincounts every label, scans bounded
rows for same-onset/same-string conflicts, and recomputes CE, pitch marginal,
conditional normalizer and the weighted objective position by position from the
saved logits. It never reads held-out test music and never writes to the cache.

```bash
python scripts/audit_losses.py --cache <cache> --tokenizer <tokenizer> \
  --checkpoint <checkpoint> --out <analysis>/loss_audit.json
```

On the previous loss-v3 cache and checkpoint 9891 this found 0 invalid labels,
388 `[UNK]` validation labels, and 46.4%/39.4% of sampled train/validation
windows containing at least one conflicting onset group. Independent
recomputation matched the vectorized objective to `2.1e-7` (CE), `1.1e-7`
(pitch with conditional normalizer) and `1.0e-7` (weighted loss), with finite
gradients. Use `--skip-scan` to skip the expensive full-split counts.

## Verification and resource probe

```bash
python -m pytest tests/ -q
# Optional independent importer cross-check; provide a local AlphaTab 1.3.0 UMD:
ALPHATAB_JS=/path/to/alphaTab.js python -m pytest tests/test_parser_independent.py -q
python scripts/benchmark_losses.py --batch-size 128 --sequence-length 512
```

Tests cover exact CE/gradient parity, pitch marginals, no mean-pitch cancellation,
normalized smoothing, protected techniques, conditioning, unavailable strings,
invalid targets, mixed-precision stability, padding, real tiny T5 train/evaluate/
reload/optimizer-state resume, fixture cache creation and both training CLI modes.
The fixture CLI tests use tiny duplicated synthetic splits only to test plumbing.

The resource probe uses synthetic targets and random model weights, never held-out
data. On the RTX 4090 at batch 128 × 512, an initial five-measurement probe after
two warmups gave approximately 0.106 s CE, 0.116 s pitch and 0.127 s smoothing per
forward/backward, with peak allocations 9.22/9.22/9.25 GB. All gradients were
finite. This is not an end-to-end throughput benchmark; smoothing currently
exceeds the epic's 10% overhead target in that probe. Re-measure on real rebuilt
data before promotion. A combined full training run has subsequently been
launched; see the linked protocol. Quality ablations and held-out test evaluation
are still pending.
