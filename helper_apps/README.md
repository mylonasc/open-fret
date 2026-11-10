# Open-Fret helper apps
#
# Small standalone tools around the core pipeline. Each helper has its own
# dependencies so the main repo stays lean.

## pair_judge — blind A/B judging of generated tabs

Serves pairs of generated tablatures (e.g. two HPO trials' predictions for
the same song) side by side with the repo's playable viewer, collects
A/B/tie/skip votes with the trial identities hidden, keeps Elo ratings per
trial in SQLite, and exports vote JSONL ready for preference-model training.

```bash
cd helper_apps/pair_judge
python -m venv /tmp/pj-venv && /tmp/pj-venv/bin/pip install -r requirements.txt
/tmp/pj-venv/bin/python server.py --trials "models/hpo-play-*/trial_*/samples" --db votes.db
# open http://127.0.0.1:8901
```

Frontend (`web/`, Vite + React 18) needs one build:

```bash
cd helper_apps/pair_judge/web && npm install && npm run build
```

See `pair_judge/README.md` for the API, the vote schema, and how results
feed hyperparameter selection and preference-model training.
