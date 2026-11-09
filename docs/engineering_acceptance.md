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

## Scope decisions

Research ablations (#14/#20 and sampling ablations in #21) are deferred per the
maintainer. #12 is explicitly untouched. Family-split engineering can proceed
without retraining or replacing the accepted current baseline; a new training
comparison is not required for this engineering pass and is not claimed.
