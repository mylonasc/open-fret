#!/bin/bash
# Playability-loss weight screen: one Optuna study per weight combination
# (loss weights are fixed per study by design). Every completed trial exports
# the fixed 5-window panel with a playable viewer, so combinations are judged
# by listening, not by scalar eval_loss. See docs/domain_aware_losses.md.
#
# Usage: scripts/run_playability_hpo.sh [--trials 2] [--time-limit 2700]
# Runs sequentially on one GPU. Resume-safe: completed studies are skipped
# (their study_summary.json exists); delete an output dir to rerun it.
# Uses $PYTHON if set, else `python` from PATH (activate the training env).
set -u
TRIALS="${1:-2}"
LIMIT="${2:-2700}"
PYBIN="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="--ce-loss-weight 0.3 --pitch-loss-alpha 1.0 --equivalence-smoothing 0.1 \
  --event-type-alpha 0.3 --conditional-pitch"
COMMON="--storage sqlite:///$ROOT/hpo-losses.db \
  --cache-dir $ROOT/dataset/processed/loss-v4-curated-20261109/cache \
  --tokenizer-dir $ROOT/dataset/processed/loss-v4-curated-20261109/tab_tokenizer \
  --sample-data-dir $ROOT/dataset/processed/loss-v4-curated-20261109/individual \
  --sample-val-files $ROOT/dataset/processed/loss-v4-curated-20261109/val_files.txt \
  --sample-songs 5 --sample-beams 2 \
  --trials $TRIALS --time-limit $LIMIT"
run_study() { # name outdir [extra loss flags...]
  local name="$1" out="$2"; shift 2
  if [ -f "$ROOT/$out/study_summary.json" ]; then
    echo "SKIP $name (study_summary.json exists)"
    return 0
  fi
  echo "=== $name ==="
  # shellcheck disable=SC2086
  "$PYBIN" "$ROOT/scripts/hpo_optuna.py" \
    --study-name "$name" --output-root "$ROOT/$out" $COMMON $BASE "$@"
}
run_study play-base models/hpo-play-base
run_study play-pos05 models/hpo-play-pos05 --position-alpha 0.5
run_study play-span05 models/hpo-play-span05 --span-alpha 0.5
run_study play-stretch05 models/hpo-play-stretch05 --stretch-alpha 0.5
run_study play-pos05-span05 models/hpo-play-pos05-span05 --position-alpha 0.5 --span-alpha 0.5
run_study play-pos05-stretch05 models/hpo-play-pos05-stretch05 --position-alpha 0.5 --stretch-alpha 0.5
run_study play-span05-stretch05 models/hpo-play-span05-stretch05 --span-alpha 0.5 --stretch-alpha 0.5
run_study play-all05 models/hpo-play-all05 --position-alpha 0.5 --span-alpha 0.5 --stretch-alpha 0.5
echo "ALL STUDIES DONE (or skipped)"
