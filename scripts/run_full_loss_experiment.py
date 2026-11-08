"""Rebuild-stage orchestration and a fresh three-epoch combined-loss run.

Requires completed AlphaTab processing in --data-root/individual. Existing
production artifacts are never overwritten; all stages have separate logs.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-name", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    data = Path(args.data_root).resolve()
    run = ROOT / "models" / args.run_name
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Fresh run requested; refusing nonempty {run}")
    from open_fret.alphatex import ALPHATAB_PARSER_VERSION
    processing = json.loads((data / "individual/_processing_manifest.json").read_text())
    if processing["parser_version"] != ALPHATAB_PARSER_VERSION or processing["windows"] <= 0:
        raise ValueError("Expected completed extended-parser processing manifest")

    def stage(name, command):
        print(f"Starting {name}: {' '.join(command)}", flush=True)
        (data / "execution_state.json").write_text(json.dumps({"stage": name, "command": command}, indent=2))
        with open(data / f"{name}.log", "w") as log:
            subprocess.run(command, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "4",
                                                  "TOKENIZERS_PARALLELISM": "false"},
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    py = sys.executable
    if not args.train_only:
        if (data / "cache").exists():
            raise FileExistsError("Cache already exists; use --train-only after verifying preparation")
        stage("split", [py, "scripts/03_split.py", "--input-dir", str(data / "individual"),
                        "--output-dir", str(data), "--seed", "42"])
        stage("vocab", [py, "scripts/01.5_generate_vocab.py", "--input-dir", str(data / "individual"),
                        "--train-list", str(data / "train_files.txt"), "--output", str(data / "unique_tokens.json"),
                        "--complete-fretboard",
                        "--structural-lists", ",".join(str(data / f) for f in
                                                       ("train_files.txt", "val_files.txt", "test_files.txt"))])
        stage("tokenize", [py, "scripts/02_tokenize.py", "--vocab", str(data / "unique_tokens.json"),
                           "--output-dir", str(data / "tab_tokenizer")])
        stage("cache", [py, "scripts/04_cache_dataset.py", "--data-dir", str(data / "individual"),
                        "--tokenizer-dir", str(data / "tab_tokenizer"), "--cache-dir", str(data / "cache"),
                        "--train-list", str(data / "train_files.txt"), "--val-list", str(data / "val_files.txt"),
                        "--test-list", str(data / "test_files.txt")])
    cache = json.loads((data / "cache/cache_manifest.json").read_text())
    if cache["skipped_or_truncated"] or not all(cache["splits"].get(s, 0) > 0 for s in ("train", "validation", "test")):
        raise ValueError("Refusing incomplete/truncated rebuilt cache")
    # Match the previous validation/checkpoint cadence as a fraction of an
    # epoch, since the scientifically supported corpus has changed size.
    eval_steps = max(1, round(math.ceil(cache["splits"]["train"] / 256) * 2000 / 5372))
    command = [py, "scripts/train_model.py", "--run-name", args.run_name,
               "--tokenizer-dir", str(data / "tab_tokenizer"), "--cache-dir", str(data / "cache"),
               "--data-dir", str(data / "individual"), "--val-files", str(data / "val_files.txt"),
               "--epochs", "3", "--batch-size", "128", "--grad-accum", "2", "--seed", "42",
               "--lr", "0.0005696488989999719", "--dropout", "0.14446443138833517",
               "--warmup-steps", "39", "--decay-start-step", "391", "--decay-factor", "0.1",
               "--eval-steps", str(eval_steps), "--save-steps", str(2 * eval_steps),
               "--sample-songs", "5", "--sample-beams", "4", "--sample-no-repeat-ngram", "3",
               "--ce-loss-weight", "0.3", "--pitch-loss-alpha", "1", "--equivalence-smoothing", "0.1",
               "--event-type-alpha", "0.3", "--conditional-pitch",
               "--optuna-storage", "sqlite:///hpo-losses.db",
               "--optuna-study-name", "full-curated-ce03-pitch1-smooth01-type03-v2"]
    (data / "full_run_plan.json").write_text(json.dumps({"fresh": True, "init_from": None,
        "resume": None, "processing": processing, "cache": cache, "command": command}, indent=2))
    if args.prepare_only:
        print(f"Preparation complete. Run plan: {data / 'full_run_plan.json'}", flush=True)
        return
    stage("train", command)
    (data / "execution_state.json").write_text(json.dumps({"stage": "complete", "run": str(run)}))


if __name__ == "__main__":
    main()
