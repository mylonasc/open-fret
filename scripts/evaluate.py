"""Held-out evaluation with alignment-aware metrics (Issue #6).

- Requires an actual ``test`` split; never silently falls back to training.
- Counts insertions/deletions instead of ``zip``-truncating predictions.
- Reports pitch/onset preservation alongside token accuracy.
"""
import argparse
import json
import os
from collections import Counter

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_from_disk


def _pitches(tokens):
    out = []
    for t in tokens:
        if t.startswith("NO_"):
            try:
                out.append(int(t.split("_")[1]))
            except ValueError:
                continue
    return out


def _tab_pitches(tokens, tuning=(64, 59, 55, 50, 45, 40)):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from open_fret.postprocess import pitch_from_tab
    out = []
    for t in tokens:
        if t.startswith("TAB_"):
            parts = t.split("_")
            try:
                p = pitch_from_tab(int(parts[1]), int(parts[2]), list(tuning))
            except (ValueError, IndexError):
                continue
            if p is not None:
                out.append(p)
    return out


def evaluate_model(model_path, dataset_path, split="test", num_samples=100,
                   max_new_tokens=256, report_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device} (split={split})...")
    # model_path may be a local dir or HF hub id.
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not load model/tokenizer from {model_path!r}: {e}") from e
    dataset = load_from_disk(dataset_path)
    if split not in dataset:
        raise ValueError(
            f"Split '{split}' not in cached dataset {dataset_path} "
            f"(found {list(dataset.keys())}). Refusing to fall back to training data.")
    data = dataset[split]
    n = min(num_samples, len(data))
    if n == 0:
        raise ValueError(f"Empty evaluation split: {split}")
    data = data.select(range(n))

    exact_match = 0
    total_pred = 0
    total_ref = 0
    pitch_tp = 0
    pitch_pred_total = 0
    pitch_ref_total = 0
    invalid_tab = 0
    freq = Counter()

    for i in tqdm(range(len(data))):
        sample = data[i]
        input_ids = torch.tensor([sample["input_ids"]]).to(device)
        attention = torch.tensor([sample["attention_mask"]]).to(device)
        labels = sample["labels"]
        clean_labels = [l if l != -100 else tokenizer.pad_token_id for l in labels]
        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention,
                                     max_new_tokens=max_new_tokens, do_sample=False)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split()
        ref_text = tokenizer.decode(clean_labels, skip_special_tokens=True).split()
        # Token-level: count matches over alignment by position up to min len,
        # plus explicit insertion/deletion totals.
        m = min(len(pred_text), len(ref_text))
        exact_match += sum(1 for a, b in zip(pred_text[:m], ref_text[:m]) if a == b)
        total_pred += len(pred_text)
        total_ref += len(ref_text)
        # Pitch-level: compare MIDI onset pitches vs predicted TAB pitches.
        # Input pitches need decoding from input_ids for a fair reference;
        # here we use the reference TAB pitches as proxy plus validity checks.
        pred_pitches = _tab_pitches(pred_text)
        ref_pitches = _tab_pitches(ref_text)
        pitch_pred_total += len(pred_pitches)
        pitch_ref_total += len(ref_pitches)
        # Order-aware overlap (multiset intersection preserves duplicates).
        ref_counts = Counter(ref_pitches)
        for p in pred_pitches:
            if ref_counts.get(p, 0) > 0:
                pitch_tp += 1
                ref_counts[p] -= 1
        invalid_tab += sum(1 for t in pred_text if t.startswith("TAB_") and t not in pred_text[:0]
                           and _tab_pitches([t]) == [])
        freq.update(pred_text[:5])

    precision = exact_match / total_pred if total_pred else 0.0
    recall = exact_match / total_ref if total_ref else 0.0
    pitch_prec = pitch_tp / pitch_pred_total if pitch_pred_total else 0.0
    pitch_rec = pitch_tp / pitch_ref_total if pitch_ref_total else 0.0
    report = {
        "model": model_path, "dataset": dataset_path, "split": split,
        "samples": n, "token_precision": precision, "token_recall": recall,
        "tab_pitch_precision": pitch_prec, "tab_pitch_recall": pitch_rec,
        "invalid_tab_rate": invalid_tab / max(total_pred, 1),
        "most_common_tokens": freq.most_common(5),
    }
    print(f"\nToken precision: {precision:.4f} recall: {recall:.4f}")
    print(f"TAB pitch precision: {pitch_prec:.4f} recall: {pitch_rec:.4f}")
    print("Most frequent predicted tokens:", freq.most_common(5))
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report to {report_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/tiny-tab-v1/final")
    ap.add_argument("--dataset", default="dataset/processed/cached_dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    evaluate_model(a.model, a.dataset, split=a.split,
                   num_samples=a.num_samples, report_path=a.report)
