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
    p.add_argument("--output-root", default="models",
                   help="Parent dir for runs. Each run gets models/<run-name>/ "
                        "unless --output-dir overrides it.")
    p.add_argument("--run-name", default=None,
                   help="Run folder name. Default: tiny-tab-YYYYMMDD-HHMMSS.")
    p.add_argument("--output-dir", default=None,
                   help="Explicit output dir (overrides --output-root/--run-name). "
                        "Prefer --run-name so runs never overwrite each other.")
    p.add_argument("--log-file", default=None,
                   help="Default: <output-dir>/train.log (kept per-run).")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Per-device batch size. 64 fits 8GB VRAM for this tiny "
                        "model; 96+ better saturates a 24GB card (see benchmark).")
    p.add_argument("--workers", type=int, default=2,
                   help="DataLoader workers (dataloader_num_workers).")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=4000,
                   help="Checkpoint every N optimizer steps (~35 min at batch 96). "
                        "Must be a round multiple of --eval-steps.")
    p.add_argument("--eval-steps", type=int, default=2000,
                   help="Validate every N steps (~2 min on 176k val).")
    p.add_argument("--save-total-limit", type=int, default=5)
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
    import datetime
    import subprocess
    args = args or parse_args()
    set_seed(args.seed)
    # Each run gets its own folder; never silently overwrite another run.
    if args.output_dir:
        output_dir = args.output_dir
    else:
        run_name = args.run_name or ("tiny-tab-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        output_dir = os.path.join(args.output_root, run_name)
    if os.path.exists(output_dir) and os.listdir(output_dir) and not args.resume:
        raise FileExistsError(
            f"Output dir {output_dir} is not empty. Pass --resume to continue it "
            "or choose another --run-name.")
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    args.log_file = args.log_file or os.path.join(output_dir, "train.log")

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
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
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

    # run.json: written BEFORE training so interrupted runs stay identifiable.
    def _git_sha():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    def _gpu_name():
        try:
            return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:
            return "unknown"

    def _cache_manifest():
        mp = os.path.join(args.cache_dir, "cache_manifest.json")
        try:
            with open(mp, encoding="utf-8") as f:
                return json.load(f)
        except OSError:
            return None

    run_meta = {
        "run_name": os.path.basename(os.path.normpath(output_dir)),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "device": _gpu_name(),
        "fp16": bool(torch.cuda.is_available()),
        "seed": args.seed,
        "tokenizer": {"dir": args.tokenizer_dir, "vocab_size": len(tokenizer)},
        "data": {"cache_dir": args.cache_dir, "train_size": len(train_dataset),
                 "val_size": len(val_dataset), "manifest": _cache_manifest()},
        "model_init": init_info,
        "hyperparams": {
            "epochs": args.epochs, "max_steps": args.max_steps,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "lr": args.lr, "warmup_steps": 1000, "dropout": args.dropout,
            "save_steps": args.save_steps, "eval_steps": args.eval_steps,
        },
        "mode": mode, "resume_checkpoint": resume_checkpoint,
        "status": "running",
    }
    with open(os.path.join(output_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("Training complete. Saving model artifacts...")
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    run_meta["status"] = "complete"
    run_meta["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_meta["result"] = {
        "train_loss": getattr(result, "training_loss", None),
        "global_steps": getattr(result, "global_step", None),
        "metrics": getattr(result, "metrics", None),
        "best_metric": trainer.state.best_metric,
    }
    with open(os.path.join(final_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    with open(os.path.join(output_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)


if __name__ == "__main__":
    train()
