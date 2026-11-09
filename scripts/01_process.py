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
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.alphatex import AlphaTexParser, PARSER_VERSION, ALPHATAB_PARSER_VERSION
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
_UPSTREAM_PARSER = None


def _process_one(args):
    global _UPSTREAM_PARSER
    file_path, output_dir, extracted_root, seed, backend = args
    version = PARSER_VERSION
    if backend == "alphatab":
        from open_fret.alphatab_parser import AlphaTabParser
        from multiprocessing.util import Finalize
        if _UPSTREAM_PARSER is None or _UPSTREAM_PARSER.process.poll() is not None:
            _UPSTREAM_PARSER = AlphaTabParser()
            Finalize(_UPSTREAM_PARSER, _UPSTREAM_PARSER.close, exitpriority=10)
        parser = _UPSTREAM_PARSER
        version = ALPHATAB_PARSER_VERSION
    else:
        parser = AlphaTexParser()
    builder = SequenceBuilder()
    windower = AlignedWindower(max_input_tokens=WINDOW_INPUT_BUDGET,
                               max_target_tokens=WINDOW_TARGET_BUDGET)
    try:
        tracks = parser.parse_file(file_path)
        if not tracks:
            return 0, f"no tracks: {file_path}", {}
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
                        "tempo": track.tempo,
                        "start_tick": w.start_tick,
                        "end_tick": w.end_tick,
                        "notes": w.notes,
                        "carry_notes": w.carry_notes,
                        "window_version": w.window_version,
                        "file": rel_path,
                        "parser_version": version,
                    }
                    with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                    out_count += 1
        return out_count, "", getattr(parser, "diagnostics", {})
    except Exception as e:  # noqa: BLE001 - counted and reported
        return 0, f"{file_path}: {e}\n{traceback.format_exc()}", {}


def main(extracted_dir="dataset/raw/extracted",
         processed_dir="dataset/processed/individual", seed=42, workers=None, parser_backend="alphatab"):
    if os.path.isdir(processed_dir) and os.listdir(processed_dir):
        raise FileExistsError("Use an empty output directory to avoid mixing stale windows")
    os.makedirs(processed_dir, exist_ok=True)
    all_files: list[str] = []
    for root, _, files in os.walk(extracted_dir):
        for name in files:
            if name.endswith(".tex"):
                all_files.append(os.path.join(root, name))
    print(f"Total files found: {len(all_files)}")
    if not all_files:
        print(f"No .tex files under {extracted_dir}; nothing to do.")
        raise ValueError("No AlphaTex input files found")
    all_files.sort()
    num_workers = workers or min(16, max(1, multiprocessing.cpu_count() // 2))
    print(f"Processing using {num_workers} workers...")
    success = 0
    failures: list[str] = []
    effects = Counter()
    skipped_unpitched = skipped_staves = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        jobs = [(f, processed_dir, extracted_dir, seed, parser_backend) for f in all_files]
        for count, err, diagnostic in tqdm(ex.map(_process_one, jobs, chunksize=16), total=len(jobs),
                               desc="Processing files"):
            success += count
            effects.update(diagnostic.get("articulations_not_targeted", {}))
            skipped_unpitched += diagnostic.get("skipped_unpitched_notes", 0)
            skipped_staves += diagnostic.get("skipped_staves", 0)
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
    with open(os.path.join(processed_dir, "_processing_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"parser_version": ALPHATAB_PARSER_VERSION if parser_backend == "alphatab" else PARSER_VERSION,
                   "seed": seed, "articulations_not_targeted": dict(effects),
                   "skipped_unpitched_notes": skipped_unpitched, "skipped_staves": skipped_staves,
                   "source_files": len(all_files), "rejected_files": len(failures),
                   "windows": success}, f, indent=2)
    if not success:
        raise ValueError("No supported windows produced; inspect _failures.log")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-dir", default="dataset/raw/extracted")
    ap.add_argument("--processed-dir", default="dataset/processed/individual")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--parser-backend", choices=["strict", "alphatab"], default="alphatab")
    a = ap.parse_args()
    main(extracted_dir=a.extracted_dir, processed_dir=a.processed_dir, seed=a.seed,
         workers=a.workers, parser_backend=a.parser_backend)
