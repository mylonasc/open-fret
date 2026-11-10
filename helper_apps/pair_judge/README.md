# Pair judge — blind A/B comparison of generated tablatures

Side-by-side playable comparison of two HPO trials' (or runs') predictions
for the same song, with trial identities hidden until you vote. Votes land in
SQLite with per-trial Elo ratings and export as JSONL preference triples for
later reward/preference-model training — and for picking the winning
hyperparameter combination by ear instead of by `eval_loss`.

## Setup

Own environment (independent of the repo venv):

```bash
cd helper_apps/pair_judge
python -m venv /tmp/pj-venv
/tmp/pj-venv/bin/pip install -r requirements.txt
cd web && npm install && npm run build && cd ..
```

`import open_fret.viewer` executes the repo `__init__`, so the env also needs
`torch` + `transformers` (CPU torch is fine — no model is ever loaded here;
only tokens are read and viewer bundles built).

## Run

```bash
/tmp/pj-venv/bin/python server.py \
  --trials "models/hpo-play-*/trial_*/samples" \
  --db votes.db --songs 5 --port 8901
# open http://127.0.0.1:8901
```

Each `--trials` value is a glob of trial sample dirs (repeatable). A trial
dir holds `manifest.json` with a `songs: [{name, file}]` list plus
`{file}.input.txt` / `{file}.pred.txt` per song — exactly what
`scripts/export_samples.py` and `scripts/train_model.py` write, so HPO
trials and training runs plug in unchanged. `--songs N` keeps the first N
songs shared by *all* trials (the fixed comparison panel). `--seed`
controls pair order and A/B side assignment.

Judge flow: pick a judge name (remembered locally), press play on either
side, vote **A better / Tie / B better** or **Skip**. Trial identities are
revealed after each vote. The Results tab shows the Elo standings (base
1500, K=24), win counts, and a JSONL download.

## API

- `GET /api/health` → `{ok, trials, pairs}`
- `GET /api/next` → `{done, pair_id, song, left: {bundle}, right: {bundle},
  total, remaining}` — bundles carry no trial labels (blind).
- `POST /api/vote {pair_id, winner: left|right|tie, notes?, judge?}` →
  `{ok, reveal: {left, right}}`; double votes rejected (409).
- `POST /api/skip {pair_id, judge?}` → advances the queue.
- `GET /api/results` → `{trials: [{trial, elo, wins, decided}], votes,
  skipped, pairs, tally}`.
- `GET /api/export` → JSONL file download, one line per decided pair.

## Data model

SQLite (`--db`, default `pair_judge.db` in CWD):

- `trials(label PRIMARY KEY, path)` — discovered sample dirs.
- `pairs(id, song, left_trial, right_trial)` — shuffled queue; sides are
  randomly flipped at build time and stored, so A/B assignment is fixed
  and auditable.
- `votes(id, pair_id UNIQUE, winner, notes, judge, ts, skipped)` — `winner`
  is `left`/`right`/`tie`; skipped pairs are recorded too (never re-served).

Elo is recomputed deterministically in `pairing.compute_elo` from votes in
rowid order.

## Export format (for preference training)

One JSON object per line:

```json
{"song": "12_Stones_-_Crash", "input_tokens": [...], "judge": "me",
 "notes": "cleaner shifts", "ts": "...", "pair_id": "p0007",
 "chosen": {"trial": "hpo-play-span05/trial_2", "tab_tokens": [...]},
 "rejected": {"trial": "hpo-play-base/trial_0", "tab_tokens": [...]}}
{"song": "...", "tie": true, "a": {...}, "b": {...}, ...}
```

Feed `chosen`/`rejected` straight into DPO-style preference training; ties
are marked so downstream code can drop them or treat them as draws. Group by
song or trial id for analysis.

## Tests

```bash
/tmp/pj-venv/bin/python -m pytest tests/ -q
```

Spins up the app in-process (TestClient) with two synthetic trial dirs:
discovery, blinding (no trial ids pre-vote), voting, ties, skips, Elo
movement, and the export schema.
