"""PyTorch dataset for aligned MIDI/tab windows.

Fixes Issue #10 items: no silent row substitution (failures raise with
file context), empty splits are rejected early, and legacy ``midi``/``tab``
keys plus new ``source`` metadata are both supported.
"""
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader


class TabDataset(Dataset):
    def __init__(self, file_list_path, data_dir, tokenizer, max_length=512,
                 required_parser_version=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.required_parser_version = required_parser_version
        if not os.path.exists(file_list_path):
            raise FileNotFoundError(f"Split file not found: {file_list_path}")
        with open(file_list_path, "r", encoding="utf-8") as f:
            self.file_names = [line.strip() for line in f if line.strip()]
        if not self.file_names:
            raise ValueError(f"Empty split file: {file_list_path}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        file_path = os.path.join(self.data_dir, name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            versions = ((self.required_parser_version,) if isinstance(self.required_parser_version, str)
                        else self.required_parser_version)
            if versions and data.get("parser_version") not in versions:
                raise ValueError("Obsolete AlphaTex supervision: reprocess into a new data directory")
            midi_tokens = data["midi"]
            tab_tokens = data["tab"]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed loading example {name}: {e}") from e
        midi_text = " ".join(midi_tokens) + " " + self.tokenizer.eos_token
        tab_text = " ".join(tab_tokens) + " " + self.tokenizer.eos_token
        inputs = self.tokenizer(
            midi_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(
            tab_text, max_length=self.max_length,
            padding="max_length", truncation=True,
            return_tensors="pt").input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        result = {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels,
        }
        if self.required_parser_version:
            from open_fret.token_metadata import MAX_STRINGS, TokenMetadata
            tuning = data.get("tuning") or TokenMetadata.tuning(midi_tokens[1])
            if tuning is None or len(tuning) > MAX_STRINGS:
                raise ValueError("Missing/unsupported tuning")
            result["loss_tuning"] = torch.tensor(list(tuning) + [-1] * (MAX_STRINGS - len(tuning)))
            result["loss_capo"] = torch.tensor(data.get("capo", TokenMetadata.capo(midi_tokens[0])))
        return result


def get_dataloader(split, data_dir, tokenizer_path, batch_size=8, max_length=512):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    file_list = f"dataset/processed/{split}_files.txt"
    dataset = TabDataset(file_list, data_dir, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


if __name__ == "__main__":
    TOKENIZER_DIR = "dataset/processed/tab_tokenizer"
    DATA_DIR = "dataset/processed/individual"
    if os.path.exists(TOKENIZER_DIR):
        try:
            loader = get_dataloader("val", DATA_DIR, TOKENIZER_DIR, batch_size=2)
            batch = next(iter(loader))
            print("Batch keys:", batch.keys())
            print("Input shape:", batch["input_ids"].shape)
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")
            raise
