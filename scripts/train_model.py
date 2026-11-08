"""First working training loop (Issues #3, #4, #5, #10).

- Fresh runs work: ``--resume`` is opt-in; without it training starts from
  scratch. ``--resume`` without a checkpoint starts fresh instead of crashing.
- Aligned windows + song-group splits are consumed here (produced upstream).
- No silent data substitution; missing artifacts fail with actionable errors.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)

from datasets import load_from_disk

from config import get_tiny_tab_config
from dataloader import TabDataset

try:
    from open_fret.training import load_init_model, resolve_resume_checkpoint
except ImportError:  # when run as scripts/train_model.py without root on path
    from open_fret.training import load_init_model, resolve_resume_checkpoint


class TrainLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "a") as f:
            f.write("\n" + "=" * 40 + "\n")
            f.write("LOG SESSION\n")
            f.write("=" * 40 + "\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch or 0)
        with open(self.log_path, "a") as f:
            f.write(f"\n--- Epoch {epoch} Summary ---\n")
            if state.log_history:
                eval_metrics = [h for h in state.log_history if "eval_loss" in h]
                if eval_metrics:
                    f.write(f"Validation Loss: {eval_metrics[-1]['eval_loss']:.4f}\n")
            f.write("-" * 20 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_path, "a") as f:
                log_str = f"Step: {state.global_step:06d} | "
                if "loss" in logs:
                    log_str += f"Train Loss: {logs['loss']:.4f} | "
                if "eval_loss" in logs:
                    log_str += f"Eval Loss: {logs['eval_loss']:.4f} | "
                if "learning_rate" in logs:
                    log_str += f"LR: {logs['learning_rate']:.2e} | "
                f.write(log_str + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Train tiny T5 tab model.")
    p.add_argument("--tokenizer-dir", default="dataset/processed/tab_tokenizer")
    p.add_argument("--cache-dir", default="dataset/processed/cached_dataset")
    p.add_argument("--data-dir", default="dataset/processed/individual")
    p.add_argument("--output-dir", default="models/tiny-tab-v1")
    p.add_argument("--log-file", default="train_log.txt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", nargs="?", const=True, default=None,
                   help="Resume from latest checkpoint in --output-dir, "
                        "or from PATH if a path is given. Omit for fresh training.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--init-from", default=None,
                   help="Optional pretrained T5 endpoint (HF hub id or local path, "
                        "e.g. 't5-small') to fine-tune instead of training from scratch.")
    p.add_argument("--dropout", type=float, default=None,
                   help="Override config dropout_rate (both scratch and pretrained).")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="If > 0, train at most this many steps (overrides --epochs). "
                        "Useful for smoke validation on the full dataset.")
    return p.parse_args()


def train(args=None):
    args = args or parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"System Check: Detected {device.upper()} for training.")

    if not os.path.isdir(args.tokenizer_dir):
        raise FileNotFoundError(
            f"Tokenizer not found at {args.tokenizer_dir}. "
            "Run 01.5_generate_vocab.py and 02_tokenize.py first.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    config = None
    model, init_info = load_init_model(
        tokenizer, init_from=args.init_from,
        tiny_config_fn=get_tiny_tab_config, dropout=args.dropout)
    print(f"Model init: {init_info}")
    if args.init_from is None and args.dropout is not None:
        print(f"Overrode tiny-tab dropout_rate to {args.dropout}")

    if os.path.exists(args.cache_dir):
        print(f"Loading high-speed cached dataset from {args.cache_dir}")
        dataset = load_from_disk(args.cache_dir)
        if "train" not in dataset or "validation" not in dataset:
            raise ValueError(
                f"Cache at {args.cache_dir} must contain 'train' and 'validation' splits, "
                f"got {list(dataset.keys())}. Delete it and re-run 04_cache_dataset.py.")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    else:
        print(f"Cache not found at {args.cache_dir}. Falling back to file-by-file loading.")
        train_dataset = TabDataset("dataset/processed/train_files.txt",
                                   args.data_dir, tokenizer, max_length=args.max_length)
        val_dataset = TabDataset("dataset/processed/val_files.txt",
                                 args.data_dir, tokenizer, max_length=args.max_length)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Empty train or validation dataset; check splits and cache.")

    # Note: Adafactor (paper standard) does not use decoupled weight decay
    # like AdamW; TrainingArguments.weight_decay is therefore 0.0 here to
    # avoid implying regularization that is not applied.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        optim="adafactor",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=10),
            TrainLoggerCallback(args.log_file),
        ],
    )

    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.resume)
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        mode = "RESUMING"
    else:
        print("Starting fresh training run (no checkpoint resume).")
        mode = "FRESH"
    with open(args.log_file, "a") as f:
        f.write(f"\n{mode} TRAINING | Target Epochs: {args.epochs} | Seed: {args.seed}\n")
        f.write("-" * 40 + "\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("Training complete. Saving model artifacts...")
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "epochs": args.epochs, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "lr": args.lr, "seed": args.seed,
            "tokenizer_dir": args.tokenizer_dir, "cache_dir": args.cache_dir,
            "mode": mode, "resume_checkpoint": resume_checkpoint,
            "init_from": args.init_from, "init_info": init_info,
            "dropout": args.dropout,
        }, f, indent=2)


if __name__ == "__main__":
    train()
