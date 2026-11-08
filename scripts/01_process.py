"""Process AlphaTex files into aligned MIDI/tab windows.

Fixes:
- Issue #2: uses :class:`open_fret.alphatex.AlphaTexParser` with explicit
  duration handling (string numbers are no longer misread as durations).
- Issue #4: emits musically aligned windows via
  :class:`open_fret.sequences.AlignedWindower` instead of whole-track
  sequences that were later truncated out of alignment.
- Issue #5: every window carries ``source`` (original .tex path) so splits
  can keep all derivatives of one song together.

Output JSON schema (per window)::

    {
      "midi": [...], "tab": [...],          # legacy keys, kept for loaders
      "source": "rel/path.tex",             # group id for splitting
      "track": 0, "window": 0, "capo": 0,
      "tuning": [...], "start_tick": 0, "end_tick": 480,
      "file": "rel/path.tex"                # legacy alias of source
    }
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.alphatex import AlphaTexParser
from open_fret.sequences import SequenceBuilder, AlignedWindower

try:
    from tqdm import tqdm
except ImportError:  # minimal fallback
    def tqdm(x, **kwargs):
        return x

MAX_INPUT_TOKENS = 512
MAX_TARGET_TOKENS = 512
# Reserve one slot for EOS appended later.
WINDOW_INPUT_BUDGET = MAX_INPUT_TOKENS - 1
WINDOW_TARGET_BUDGET = MAX_TARGET_TOKENS - 1


def _process_one(args):
    file_path, output_dir, extracted_root, seed = args
    parser = AlphaTexParser()
    builder = SequenceBuilder()
    windower = AlignedWindower(max_input_tokens=WINDOW_INPUT_BUDGET,
                               max_target_tokens=WINDOW_TARGET_BUDGET)
    try:
        tracks = parser.parse_file(file_path)
        if not tracks:
            return 0, f"no tracks: {file_path}"
        rel_path = os.path.relpath(file_path, extracted_root)
        rng = random.Random(f"{seed}:{rel_path}")
        out_count = 0
        for track_idx, track in enumerate(tracks):
            if not track.events:
                continue
            for capo in (0, rng.randint(1, 7)):
                steps, tuning = builder.build_steps(track, capo=capo)
                if not steps:
                    continue
                windows = windower.window(steps, tuning, capo=capo,
                                          source=rel_path, track_index=track_idx)
                for w in windows:
                    if not w.midi_tokens or not w.tab_tokens:
                        continue
                    safe = rel_path.replace(os.sep, "__")
                    # Include track/capo/window to avoid collisions.
                    name = f"{safe}__t{track_idx}__c{capo}__w{w.window_index}.json"
                    payload = {
                        "midi": w.midi_tokens,
                        "tab": w.tab_tokens,
                        "source": rel_path,
                        "track": track_idx,
                        "window": w.window_index,
                        "capo": capo,
                        "tuning": w.tuning,
                        "start_tick": w.start_tick,
                        "end_tick": w.end_tick,
                        "file": rel_path,
                    }
                    with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                    out_count += 1
        return out_count, ""
    except Exception as e:  # noqa: BLE001 - counted and reported
        return 0, f"{file_path}: {e}\n{traceback.format_exc()}"


def main(extracted_dir="dataset/raw/extracted",
         processed_dir="dataset/processed/individual", seed=42):
    os.makedirs(processed_dir, exist_ok=True)
    all_files: list[str] = []
    for root, _, files in os.walk(extracted_dir):
        for name in files:
            if name.endswith(".tex"):
                all_files.append(os.path.join(root, name))
    print(f"Total files found: {len(all_files)}")
    if not all_files:
        print(f"No .tex files under {extracted_dir}; nothing to do.")
        return
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    print(f"Processing using {num_workers} workers...")
    success = 0
    failures: list[str] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        jobs = [(f, processed_dir, extracted_dir, seed) for f in all_files]
        for count, err in tqdm(ex.map(_process_one, jobs), total=len(jobs),
                               desc="Processing files"):
            success += count
            if err:
                failures.append(err)
    print(f"\nFinished! Successfully wrote {success} windows.")
    print(f"Individual JSON files saved at: {processed_dir}")
    if failures:
        print(f"{len(failures)} files produced no output (showing up to 10):")
        for err in failures[:10]:
            print(" -", err.splitlines()[0])
        with open(os.path.join(processed_dir, "_failures.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(failures))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-dir", default="dataset/raw/extracted")
    ap.add_argument("--processed-dir", default="dataset/processed/individual")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(extracted_dir=a.extracted_dir, processed_dir=a.processed_dir, seed=a.seed)
