"""Cache aligned windows for training (Issue #4 follow-through).

Windows are already musically aligned upstream; this step validates lengths,
tokenizes, masks padding with -100, caches train/validation/test, and writes
a manifest with tokenizer + data versions. Failures are counted, not silent.
"""
import json
import multiprocessing
import os

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


def create_cache(tokenizer_dir="dataset/processed/tab_tokenizer",
                 data_dir="dataset/processed/individual",
                 train_list="dataset/processed/train_files.txt",
                 val_list="dataset/processed/val_files.txt",
                 test_list="dataset/processed/test_files.txt",
                 cache_dir="dataset/processed/cached_dataset",
                 max_length=512):
    print("Initializing dataset caching...")
    if not os.path.exists(tokenizer_dir):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_dir}. Run 02_tokenize.py first.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    skipped = {"count": 0}

    def process_file_list(file_list_path, split):
        if not os.path.exists(file_list_path):
            if split in ("train", "validation"):
                raise FileNotFoundError(f"Required split file missing: {file_list_path}")
            print(f"Optional split file missing, skipping: {file_list_path}")
            return None
        with open(file_list_path, "r", encoding="utf-8") as f:
            file_names = [ln.strip() for ln in f if ln.strip()]
        print(f"Loading and tokenizing {len(file_names)} files from {file_list_path}...")

        def gen():
            for fname in tqdm(file_names, desc=f"Reading {split}"):
                fpath = os.path.join(data_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    midi_tokens, tab_tokens = data["midi"], data["tab"]
                    # Upstream windowing reserves EOS; warn if truncation would occur.
                    if len(midi_tokens) + 1 > max_length or len(tab_tokens) + 1 > max_length:
                        skipped["count"] += 1
                    yield {"midi_text": " ".join(midi_tokens) + " " + tokenizer.eos_token,
                           "tab_text": " ".join(tab_tokens) + " " + tokenizer.eos_token}
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
        model_inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in model_inputs["labels"]
        ]
        return model_inputs

    print("Converting to model-ready tensors (parallel map)...")
    num_cpus = multiprocessing.cpu_count()
    num_proc = num_cpus if os.name != "nt" else 1
    tokenized_train = train_ds.map(tokenize_function, batched=True,
                                   num_proc=num_proc, remove_columns=["midi_text", "tab_text"])
    tokenized_val = val_ds.map(tokenize_function, batched=True,
                               num_proc=num_proc, remove_columns=["midi_text", "tab_text"])
    splits = {"train": tokenized_train, "validation": tokenized_val}
    if test_ds is not None:
        splits["test"] = test_ds.map(tokenize_function, batched=True,
                                     num_proc=num_proc, remove_columns=["midi_text", "tab_text"])
    ds_dict = DatasetDict(splits)
    print(f"Saving cached dataset to {cache_dir}...")
    ds_dict.save_to_disk(cache_dir)
    with open(os.path.join(cache_dir, "cache_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_dir": tokenizer_dir, "max_length": max_length,
                   "skipped_or_truncated": skipped["count"],
                   "splits": {k: len(v) for k, v in splits.items()}}, f, indent=2)
    print(f"Caching complete. Skipped/truncated: {skipped['count']}.")
    if skipped["count"]:
        print("WARNING: some windows needed truncation or were skipped; "
              "check windowing budgets if this count is large.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", default="dataset/processed/tab_tokenizer")
    ap.add_argument("--data-dir", default="dataset/processed/individual")
    ap.add_argument("--cache-dir", default="dataset/processed/cached_dataset")
    a = ap.parse_args()
    create_cache(tokenizer_dir=a.tokenizer_dir, data_dir=a.data_dir,
                 train_list="dataset/processed/train_files.txt",
                 val_list="dataset/processed/val_files.txt",
                 test_list="dataset/processed/test_files.txt",
                 cache_dir=a.cache_dir)
