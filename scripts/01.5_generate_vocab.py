"""Build the unique-token list.

By default scans all windows, but for leakage-free runs pass
``--train-list dataset/processed/train_files.txt`` to build the vocabulary
from training windows only (requires splitting before vocab generation).
"""
import argparse
import json
import os
from tqdm import tqdm


def generate_vocab(input_dir="dataset/processed/individual",
                    output_file="dataset/processed/unique_tokens.json",
                    train_list=None, complete_fretboard=False,
                    structural_lists=None):
    """Build the token vocabulary.

    ``train_list`` restricts learned tokens to training windows. ``structural_lists``
    may point at all splits: CAPO/TUNING/NO/NF/TS are inference inputs or timing
    structure, not melodic targets, so including them is not target leakage and
    avoids silently encoding validation conditioning as ``[UNK]``. TAB tokens
    remain train-only.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory {input_dir} not found. Run 01_process.py first.")
    if train_list:
        with open(train_list, encoding="utf-8") as f:
            files = [ln.strip() for ln in f if ln.strip()]
        print(f"Building training-only vocab from {len(files)} files (leakage-safe).")
    else:
        files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        print(f"Collecting unique tokens from {len(files)} files...")
        print("WARNING: vocab includes validation/test tokens. "
              "For final runs, use --train-list train_files.txt after splitting.")
    structural_files = []
    for path in structural_lists or []:
        with open(path, encoding="utf-8") as f:
            structural_files.extend(ln.strip() for ln in f if ln.strip())
    unique_tokens = set()
    if complete_fretboard:
        # Predefined domain tokens, NOT vocabulary learned from held-out targets.
        unique_tokens.update(f"TAB_{s}_{f}" for s in range(1, 13) for f in range(25))
        unique_tokens.update(f"{kind}_{p}" for kind in ("NO", "NF") for p in range(128))
        unique_tokens.update(f"CAPO_{c}" for c in range(8))
        unique_tokens.update(f"TS_{t}" for t in range(1, 1921))
    for filename in tqdm(files, desc="Scanning files"):
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "midi" in data:
                unique_tokens.update(data["midi"])
            if "tab" in data:
                unique_tokens.update(data["tab"])
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {filename}: {e}")
            continue
    structural_prefixes = ("CAPO_", "TUNING_", "NO_", "NF_", "TS_")
    already_scanned = set(files)
    for filename in tqdm(structural_files, desc="Scanning split conditioning"):
        if filename in already_scanned:
            continue
        try:
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"Skipping structural token source {filename}: {e}")
            continue
        for stream in ("midi", "tab"):
            unique_tokens.update(t for t in data.get(stream, [])
                                 if t.startswith(structural_prefixes))
    sorted_tokens = sorted(unique_tokens)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_tokens, f)
    print(f"Done! Saved {len(sorted_tokens)} unique tokens to {output_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="dataset/processed/individual")
    ap.add_argument("--output", default="dataset/processed/unique_tokens.json")
    ap.add_argument("--train-list", default=None)
    ap.add_argument("--complete-fretboard", action="store_true")
    ap.add_argument("--structural-lists", default=None,
                    help="Comma-separated split lists whose CAPO/TUNING/NO/NF/TS "
                         "tokens are included without using their TAB targets.")
    a = ap.parse_args()
    lists = [p for p in (a.structural_lists or "").split(",") if p]
    generate_vocab(a.input_dir, a.output, a.train_list,
                   complete_fretboard=a.complete_fretboard, structural_lists=lists)
