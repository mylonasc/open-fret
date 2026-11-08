"""Synthetic forward/backward resource probe; NOT a model-quality evaluation."""

import argparse
import contextlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from open_fret.losses import LossConfig, TabLoss
from open_fret.token_metadata import TokenMetadata
from scripts.config import get_tiny_tab_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", default="dataset/processed/tab_tokenizer")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.iterations) < 1 or args.sequence_length < 4 or args.warmup < 0:
        parser.error("Positive batch/iterations, sequence length >= 4 and warmup >= 0 required")
    torch.manual_seed(42)
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    metadata = TokenMetadata.from_tokenizer(tok)
    required = ["CAPO_0", "TUNING_64_59_55_50_45_40"]
    if any(t not in tok.get_vocab() for t in required):
        parser.error("Synthetic probe requires standard tuning and capo-zero vocabulary tokens")
    note_ids = [i for i in range(len(tok)) if (n := metadata.note(i))
                and 1 <= n[0] <= 6 and n[1] <= 24 and n[2] is None]
    if not note_ids:
        parser.error("No valid plain TAB tokens in vocabulary")
    ids = torch.tensor(note_ids, device=device)
    shape = (args.batch_size, args.sequence_length)
    labels = ids[torch.randint(len(ids), shape, device=device)]
    labels[:, -1] = tok.eos_token_id
    inputs = torch.full(shape, tok.pad_token_id, device=device, dtype=torch.long)
    inputs[:, 0] = tok.convert_tokens_to_ids(required[0])
    inputs[:, 1] = tok.convert_tokens_to_ids(required[1])
    inputs[:, 2:] = tok.eos_token_id
    model = T5ForConditionalGeneration(get_tiny_tab_config(len(tok))).to(device)
    results = []
    for config in [LossConfig(), LossConfig(pitch_alpha=0.3), LossConfig(smoothing=0.1),
                   LossConfig(ce_weight=0.3, pitch_alpha=1, smoothing=0.1)]:
        objective = TabLoss(metadata, config).to(device)
        times, peaks = [], []
        for step in range(args.warmup + args.iterations):
            model.zero_grad(set_to_none=True)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16) if cuda else contextlib.nullcontext():
                outputs = model(input_ids=inputs, attention_mask=inputs.ne(tok.pad_token_id),
                                decoder_input_ids=model.prepare_decoder_input_ids_from_labels(labels))
                loss, _ = objective(outputs.logits, labels, inputs)
            loss.backward()
            if cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= args.warmup:
                times.append(elapsed)
                if cuda:
                    peaks.append(torch.cuda.max_memory_allocated() / 1e9)
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all()
                                              for p in model.parameters() if p.grad is not None):
                raise RuntimeError("Nonfinite loss/gradients in resource probe")
            del outputs, loss
        results.append({"loss": config.to_dict(), "mean_step_seconds": sum(times) / len(times),
                        "peak_allocated_gb": max(peaks) if peaks else None})
    print(json.dumps({"synthetic_only": True, "device": device, "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0) if cuda else None,
                      "batch_size": args.batch_size, "sequence_length": args.sequence_length,
                      "iterations": args.iterations, "token_metadata": metadata.manifest(),
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
