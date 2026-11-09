# Engineering acceptance workflow

## Reproducible CPU setup (Linux, Python 3.12, Node 22)

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
npm ci --prefix tools/alphatex_parser
OMP_NUM_THREADS=2 python -m pytest tests/ -q
python scripts/run_pipeline.py --fixture --workspace /path/to/new-fixture-workspace
```

Direct runtime/test dependencies and the AlphaTab importer are pinned. Preserve
`pip freeze` with deployment artifacts for an exact transitive environment lock.
The GitHub CPU acceptance workflow runs tests and the real synthetic fixture;
it never downloads the corpus or uses production training artifacts.

The fixture parses ten independent synthetic songs, creates a family-disjoint
8/1/1 split, caches all three splits without truncation, trains from scratch,
resumes, reloads and decodes a six-window MIDI. An independent AlphaTab import
asserts every expected start/end/pitch. `acceptance.json` is published only after
all assertions pass; mere directory existence does not indicate completion.
Synthetic scores are plumbing checks, not a quality evaluation.

Optional browser check (Playwright + Chromium/Chrome installed):

```bash
python ui/tab_data_viewer/check_completeness.py --file /path/to/new-fixture-workspace/viewer.html
```

This asserts the browser-imported score's complete note starts/ends/pitches,
toggle operation, seek preservation and advancing playback. The new page is
separate from original checkpoint audit previews.

## Complete-song inference and duration policy

```bash
python scripts/inference.py --input song.mid --model /path/to/local/run/final \
  --out corrected.tokens --raw-out raw.tokens --report inference.json \
  --tex-out song.tex --ascii-out song.txt
```

- Source notes retain stable track/channel/attack identities and absolute ticks
  at 480 PPQ, including initial silence. Equal-pitch releases use FIFO within
  track/channel/pitch; MIDI cannot encode a nested note identity on release.
- Non-drum instruments are indexed by track/channel/program. Multiple pitched
  instruments require `--instrument INDEX`; channel 10 is ignored and reported.
  Variable tempo, pedal sustain, pitch bends, malformed releases and zero-tick
  durations fail explicitly. Tick conversion rounds to nearest 480-PPQ tick.
- Chunking carries the connecting TS with its following event group, reserves
  prefix/EOS budgets and never splits simultaneous attacks. Source intervals
  live outside model context and remain intact across every chunk boundary.
- Raw anchored output enforces pitch/attack/onset and same-onset string legality.
  The separately reported corrector matches **exact onset ticks**, inserts/drops
  missing/extra notes explicitly, and searches a global legal string/fret
  assignment with occupied strings held through each source note's end. It can
  backtrack/refinger preceding notes. It fails without partial output if inputs
  are impossible or its bounded search is exhausted. This is not a general
  ergonomic/biomechanical guarantee. Corrected metrics are not raw model gains.
- `--capo N --tuning 64 59 55 50 45 38` use concert MIDI pitches highest-string
  first; capo is added exactly once. Rendering applies the same conditioning.
- AlphaTex with identity-bearing intervals preserves independent releases and
  silence using individual ties, including overlapping voices and bar crossings.
  Unrepresentable tick rhythms and unsupported effects raise rather than silently
  dropping notes. Token-only rendering remains an explicitly different policy:
  each onset sustains to the next onset/final TS (480 if no final duration).
  ASCII is an aligned onset preview, not independent-duration notation.

## Data and pipeline boundaries

New processed windows use `aligned-v2-identities`. Each attack belongs to exactly
one window; `start_tick` restores its absolute time, `notes` retains original
identities/ends, and `carry_notes` lists intervals already sounding at the window
origin. Carried notes are **not reattacked or resupervised**. Note-off-only windows
are legal. Oversized single steps fail. Cached and uncached paths use the same
strict encoder (including curation masks, padding and EOS); neither truncates.
Cache manifests include data/tokenizer hashes and row-ordered `*.sources.jsonl`
sidecars identify the original JSON rows and their content hashes.

```bash
./run_pipeline.sh --workspace /path/to/new-versioned-data --run-name new-baseline
```

Full pipeline stages are atomically published and content/config-version checked
on restart. Missing completion manifests, changed inputs/code, changed output
content and failed/partial stages fail visibly; use a new workspace rather than
deleting or overwriting existing data. Existing legacy caches are not migrated.
The raw archive is CRC checked and extracted atomically after path safety checks.

New split code groups named song/version families and **full-source** exact
musical-event fingerprints (including timing/multiplicity); it does not merge
unrelated songs by common short riffs. All derivatives are assigned to their
family and primary validation/test exclude nonzero synthetic capo derivatives.
Manifests include identities, fingerprints, group/source/track/window counts and
excluded derivative counts. Existing checkpoints and frozen panels are untouched:
new splits require separate retraining before any generalization comparison.

Processing now emits unaugmented windows only. `scripts/05_augment_train.py`
runs **after** the family split, creates deterministic source-level capo choices
only for training, and copies primary validation/test unchanged. The versioned
pipeline and CPU fixture both execute this ordering; validation/test derivatives
are never generated in this stage. Model tokens remain backward compatible.

Parser metadata includes original track name, program, string count and voice
indices/counts. Source articulation projections are explicitly recorded rather
than advertised as full performance-effect support. The default declared domain
is `mixed`; use `--instrument-policy guitar` on processing/the full pipeline for
programs 24–31 with 6–8 strings. Bass and unknown instruments are distinct domains,
with rejection/projection counts in the processing manifest. This is a dataset
policy applied before splitting, never a hidden validation filter. Voice indices
are preserved; no inferred melodic/accompaniment role annotations are claimed.

`train_model.py --sampling-strategy source` is an opt-in alternative to ordinary
window sampling: every source has equal sampling mass, independent of its number
of windows, with replacement and fixed epoch length. Sampling is epoch-seeded;
resume verifies the sampling policy and row-ordered source identities. This
requires new cache provenance and leaves ordinary training unchanged. Its
comparison against window sampling is a **deferred research ablation**, not a
claimed quality gain.

## Evaluation and frozen validation panels

```bash
# Use a NEW v2 panel/report path; never replace historical audit evidence.
python scripts/validate_generation.py --checkpoint /path/to/frozen/checkpoint \
  --tokenizer /path/to/tokenizer --data-root /path/to/processed \
  --sources 512 --decode-n 512 --manifest /path/to/new-panel-v2.json \
  --out /path/to/new-validation-report.json
# Diagnostic subsets may use --decode-n 32; their unattempted rows are explicit.
python scripts/evaluate.py --model /path/to/frozen/local/artifact \
  --dataset /path/to/strict-cache --split validation --report validation.json
```

The evaluator requires the requested validation/test split and row-ordered
unmasked source provenance; it never substitutes training data or treats masked
training labels as the original reference. Source hashes are checked. Old caches
without provenance require a separate rebuild; no old artifacts are overwritten.
Model/config/tokenizer/cache identities and deterministic decoder settings are
recorded in machine-readable reports.

Validation panel selection is model-independent: one uniform representative per
source, coverage floors over observed style/tuning/capo/domain/length/reference-
fret strata, then uniform remaining sources. There is no feasibility filtering.
This balanced diagnostic panel is **not** an unbiased population-quality estimate.
Arpeggio labels are explicitly overlapping-release/monophonic-attack proxies.
The manifest freezes split identity, rows and content hashes before generation.
Changed input/schema/selection parameters require a new manifest path.

Reports distinguish unconstrained legacy output (no repetition ban), anchored
output, corrected output and a no-learning minimum-fret/onset-duration baseline.
One-to-one timed matching preserves multiplicity; missing/extra/shifted notes and
truncated outputs reduce scores. Reports include timing error/tolerance in ticks
and tempo-qualified seconds, edit distance, valid fingering/conflict counts,
temporal coverage, EOS/cap stops and exact-reference diagnostics. Failed sources
remain in zero-penalized macro-F1 denominators; success-only metrics are labeled
separately. Source strata and heuristic family-cluster bootstrap intervals include
support/attempt/failure counts. Carry-in windows cannot claim full sustained
occupancy correction without complete-song context and report that limitation.

Five previews remain qualitative. Anchored F1 measures input preservation, not
learned fingering improvement. All checks in this engineering pass use synthetic
validation data; the real held-out test has not been evaluated or used for tuning.

### Parser acceptance evidence

Strict and pinned AlphaTab fixtures cover the original `:4 0.1 0.2 0.6`
regression, explicit/inherited/dotted durations, chords/rests, concert tuning,
techniques, tuplets/ties and unsupported-syntax/tempo errors. The already-existing
curated corpus manifest records 47,299 source files, 13,035 rejected files and
1,054,200 windows from `alphatab-1.3.0-midi-duration-v1`; articulation projections
are counted there. These are historical data-processing counts, not model
quality scores. The current work preserves that evidence and does not rebuild
the accepted training corpus in place.

## Scope decisions

Research ablations (#14/#20 and sampling ablations in #21) are deferred per the
maintainer. #12 is explicitly untouched. Family-split engineering can proceed
without retraining or replacing the accepted current baseline; a new training
comparison is not required for this engineering pass and is not claimed.

## Optuna engineering acceptance

```bash
python scripts/hpo_optuna.py --fixture-space --trials 2 --max-steps 2 --eval-steps 1 \
  --pruner none --gpu-monitor-interval 0 --study-name synthetic-acceptance \
  --storage sqlite:///fixture-study.db --output-root /path/to/new-hpo-output \
  --cache-dir /path/to/fixture/cache --tokenizer-dir /path/to/fixture/tokenizer
# Repeat against the same storage/name to resume; --trials is additional trials.
# Production search omits --fixture-space and uses explicitly bounded budgets.
python scripts/train_model.py --hpo-config /path/to/new-hpo-output/best_config.json \
  --run-name chosen-config --tokenizer-dir /path/to/tokenizer --cache-dir /path/to/cache
```

The real CPU integration test runs two distinct fresh-model trials against a
persistent SQLite study, checks intermediate reporting, resumes a third trial
with a threshold pruner, reloads the winning artifacts and passes the exported
configuration directly into normal training. Explicit training CLI flags override
the export. Study metadata records data/tokenizer identity, direction, seeds,
loss components, failures/pruning and step/epoch/time budgets. Changed data/init
or search protocols require a different study name. A per-device advisory lock
prevents parallel HPO processes in this user account; unrelated external GPU jobs
must still be coordinated by the operator. No production HPO or retraining was
launched for this check.

Final scientific ablations/comparisons remain deferred; held-out final reporting
and publication are not implied by passing this fixture.

### Recorded verification

On November 9, a **new empty Python 3.12 virtualenv** installed the documented CPU
wheel and pinned requirements, passed `pip check`, and passed **153 tests**. The
complete fixture also passed in that clean environment, including independent
12-note/six-window AlphaTab import. Browser completeness/playback was checked on
the separately generated fixture. These checks do not claim a remote CI run has
finished; the committed workflow executes the same acceptance commands.

The subsequent data-policy/evaluation batch passes **164 tests** in that same
clean environment. The expanded fixture also passed post-split training-only
augmentation (16 train / 1 validation / 1 test synthetic windows), resume/reload,
six-window MIDI and exact independent AlphaTab import. No production HPO,
retraining, held-out scoring or model release was performed.

Remote CI then exposed a hidden corpus dependency in two viewer tests. That was
fixed rather than skipping those tests. A `git archive` export of `1e94733`, with
**no dataset or model directories**, installed only the pinned Node parser and
passed **165 tests** plus the full CPU fixture in the fresh Python environment.
The separately generated browser page again imported all 12 exact expected notes
and passed toggle/seek/playback checks with no page errors.
[GitHub run 42683709667](https://github.com/mylonasc/open-fret/actions/runs/42683709667)
also passed the corpus-free check, tests, full fixture and artifact upload.
Pipeline input identities include Python sources, requirements, the Node parser
worker and its lockfile so parser/dependency changes cannot silently reuse stages.
