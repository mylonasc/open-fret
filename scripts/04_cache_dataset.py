"""Cache aligned windows for training (Issue #4 follow-through).

Windows are already musically aligned upstream; this step validates lengths,
tokenizes, masks padding with -100, caches train/validation/test, and writes
a manifest with tokenizer + data versions. Failures are counted, not silent.
"""
import json
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.alphatex import PARSER_VERSION, SUPPORTED_PARSER_VERSIONS, StaleParserDataError
from open_fret.curation import conflict_positions
from open_fret.token_metadata import MAX_STRINGS, TokenMetadata

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


def create_cache(tokenizer_dir="dataset/processed/tab_tokenizer",
                 data_dir="dataset/processed/individual",
                 train_list="dataset/processed/train_files.txt",
                 val_list="dataset/processed/val_files.txt",
                 test_list="dataset/processed/test_files.txt",
                 cache_dir="dataset/processed/cached_dataset",
                 max_length=512, mask_conflicts=True):
    print("Initializing dataset caching...")
    if not os.path.exists(tokenizer_dir):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_dir}. Run 02_tokenize.py first.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    vocab = tokenizer.get_vocab()
    skipped = {"count": 0}
    curation = {"enabled": mask_conflicts, "totals": {"masked_windows": 0,
                "masked_notes": 0, "scanned_windows": 0}, "splits": {}}
    with open(train_list, encoding="utf-8") as f:
        first = next(line.strip() for line in f if line.strip())
    with open(os.path.join(data_dir, first), encoding="utf-8") as f:
        parser_version = json.load(f).get("parser_version")

    def process_file_list(file_list_path, split):
        if not os.path.exists(file_list_path):
            if split in ("train", "validation"):
                raise FileNotFoundError(f"Required split file missing: {file_list_path}")
            print(f"Optional split file missing, skipping: {file_list_path}")
            return None
        stats = curation["splits"].setdefault(split, {"masked_windows": 0,
                                                      "masked_notes": 0,
                                                      "scanned_windows": 0})
        with open(file_list_path, "r", encoding="utf-8") as f:
            file_names = [ln.strip() for ln in f if ln.strip()]
        print(f"Loading and tokenizing {len(file_names)} files from {file_list_path}...")

        def gen():
            for fname in tqdm(file_names, desc=f"Reading {split}"):
                fpath = os.path.join(data_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("parser_version") not in SUPPORTED_PARSER_VERSIONS:
                        raise StaleParserDataError(f"Obsolete parser_version in {fname}; reprocess into a new directory")
                    if data["parser_version"] != parser_version:
                        raise StaleParserDataError("Mixed parser versions; rebuild in an empty directory")
                    midi_tokens, tab_tokens = data["midi"], data["tab"]
                    unknown = [t for t in midi_tokens + tab_tokens if t not in vocab]
                    if unknown:
                        raise ValueError(
                            "Tokens missing from vocabulary (would become [UNK]); "
                            f"rebuild the tokenizer with structural tokens: {unknown[:3]}")
                    # Upstream windowing reserves EOS; warn if truncation would occur.
                    if len(midi_tokens) + 1 > max_length or len(tab_tokens) + 1 > max_length:
                        skipped["count"] += 1
                    tuning = data.get("tuning") or TokenMetadata.tuning(midi_tokens[1])
                    capo = data.get("capo", TokenMetadata.capo(midi_tokens[0]))
                    if tuning is None or capo is None or len(tuning) > MAX_STRINGS:
                        raise ValueError("Missing/unsupported source conditioning")
                    keep = [1] * len(tab_tokens)
                    if mask_conflicts:
                        ignored = conflict_positions(tab_tokens)
                        for index in ignored:
                            keep[index] = 0
                        for target in (stats, curation["totals"]):
                            target["scanned_windows"] += 1
                            target["masked_windows"] += bool(ignored)
                            target["masked_notes"] += len(ignored)
                    yield {"midi_text": " ".join(midi_tokens) + " " + tokenizer.eos_token,
                           "tab_text": " ".join(tab_tokens) + " " + tokenizer.eos_token,
                           "label_keep": keep + [1],
                           "loss_tuning": list(tuning) + [-1] * (MAX_STRINGS - len(tuning)),
                           "loss_capo": capo}
                except StaleParserDataError:
                    raise
                except Exception as e:  # noqa: BLE001
                    skipped["count"] += 1
                    print(f"Skipping {fname}: {e}")
                    continue
        return Dataset.from_generator(gen)

    train_ds = process_file_list(train_list, "train")
    val_ds = process_file_list(val_list, "validation")
    test_ds = process_file_list(test_list, "test")

    def tokenize_function(examples):
        model_inputs = tokenizer(
            text=examples["midi_text"], text_target=examples["tab_text"],
            max_length=max_length, truncation=True, padding="max_length")
        labels = model_inputs["labels"]
        for row, keep in enumerate(examples["label_keep"]):
            if len(keep) > len(labels[row]):
                raise ValueError("Target was truncated before its mask could be applied")
            for index, value in enumerate(keep):
                if not value:
                    labels[row][index] = -100
        model_inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in labels
        ]
        model_inputs["loss_tuning"] = examples["loss_tuning"]
        model_inputs["loss_capo"] = examples["loss_capo"]
        return model_inputs

    print("Converting to model-ready tensors (parallel map)...")
    num_cpus = multiprocessing.cpu_count()
    num_proc = min(8, num_cpus) if os.name != "nt" else 1
    tokenized_train = train_ds.map(tokenize_function, batched=True,
                                   num_proc=num_proc, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(tokenize_function, batched=True,
                                num_proc=num_proc, remove_columns=val_ds.column_names)
    splits = {"train": tokenized_train, "validation": tokenized_val}
    if test_ds is not None:
        splits["test"] = test_ds.map(tokenize_function, batched=True,
                                     num_proc=num_proc, remove_columns=test_ds.column_names)
    ds_dict = DatasetDict(splits)
    print(f"Saving cached dataset to {cache_dir}...")
    ds_dict.save_to_disk(cache_dir)
    with open(os.path.join(cache_dir, "cache_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_dir": tokenizer_dir, "max_length": max_length,
                   "parser_version": parser_version, "mask_conflicts": mask_conflicts,
                   "curation": curation,
                   "skipped_or_truncated": skipped["count"],
                   "splits": {k: len(v) for k, v in splits.items()}}, f, indent=2)
    print(f"Caching complete. Skipped/truncated: {skipped['count']}, "
          f"masked conflicting notes: {curation['totals']['masked_notes']} "
          f"in {curation['totals']['masked_windows']} windows.")
    if skipped["count"]:
        print("WARNING: some windows needed truncation or were skipped; "
              "check windowing budgets if this count is large.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", default="dataset/processed/tab_tokenizer")
    ap.add_argument("--data-dir", default="dataset/processed/individual")
    ap.add_argument("--cache-dir", default="dataset/processed/cached_dataset")
    ap.add_argument("--train-list", default="dataset/processed/train_files.txt")
    ap.add_argument("--val-list", default="dataset/processed/val_files.txt")
    ap.add_argument("--test-list", default="dataset/processed/test_files.txt",
                    help="Missing/empty test list is skipped (dev subsets).")
    ap.add_argument("--no-mask-conflicts", action="store_true",
                    help="Keep same-onset/same-string reference notes (legacy caches only).")
    a = ap.parse_args()
    create_cache(tokenizer_dir=a.tokenizer_dir, data_dir=a.data_dir,
                 train_list=a.train_list, val_list=a.val_list,
                 test_list=a.test_list, cache_dir=a.cache_dir,
                 mask_conflicts=not a.no_mask_conflicts)
