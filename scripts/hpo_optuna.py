"""Optuna hyperparameter search (Issue #11).

Searches a bounded space on the fixed validation split, with pruning,
resumable storage, and export of the winning config for final retraining.
Test data must never be used here; pass only train/validation artifacts.

Example:
    python scripts/hpo_optuna.py --trials 20 --study-name open_fret_v1 \\
        --storage sqlite:///hpo.db --tokenizer-dir dataset/processed/tab_tokenizer
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPACE = {
    "lr": (1e-5, 1e-3, True),
    "dropout": (0.0, 0.2),
    "warmup_steps": (100, 2000),
    "batch_size": [8, 16, 32],
    "grad_accum": [1, 2, 4],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--study-name", default="open_fret_hpo")
    p.add_argument("--storage", default="sqlite:///hpo.db")
    p.add_argument("--tokenizer-dir", default="dataset/processed/tab_tokenizer")
    p.add_argument("--cache-dir", default="dataset/processed/cached_dataset")
    p.add_argument("--output-root", default="models/hpo")
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def suggest(trial):
    import optuna
    _ = optuna  # for linters
    lr_low, lr_high, log = SPACE["lr"]
    d_low, d_high = SPACE["dropout"]
    w_low, w_high = SPACE["warmup_steps"]
    return {
        "lr": trial.suggest_float("lr", lr_low, lr_high, log=log),
        "dropout": trial.suggest_float("dropout", d_low, d_high),
        "warmup_steps": trial.suggest_int("warmup_steps", w_low, w_high),
        "batch_size": trial.suggest_categorical("batch_size", SPACE["batch_size"]),
        "grad_accum": trial.suggest_categorical("grad_accum", SPACE["grad_accum"]),
    }


def objective(trial, args):
    import torch
    from transformers import (T5ForConditionalGeneration, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)
    from datasets import load_from_disk
    from config import get_tiny_tab_config
    import optuna

    params = suggest(trial)
    set_seed(args.seed + trial.number)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    config = get_tiny_tab_config(vocab_size=len(tokenizer))
    config.dropout_rate = params["dropout"]
    model = T5ForConditionalGeneration(config)
    dataset = load_from_disk(args.cache_dir)
    if "validation" not in dataset:
        raise ValueError("HPO requires a validation split in the cached dataset.")
    # Small subset for HPO speed; full data is used for final retraining.
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    out_dir = os.path.join(args.output_root, f"trial_{trial.number}")
    targs = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["grad_accum"],
        num_train_epochs=args.max_epochs, learning_rate=params["lr"],
        weight_decay=0.0, warmup_steps=params["warmup_steps"],
        lr_scheduler_type="cosine", optim="adafactor",
        eval_strategy="epoch", save_strategy="no",
        logging_strategy="no", fp16=torch.cuda.is_available(),
        report_to="none", seed=args.seed + trial.number)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds)
    trainer.train()
    metrics = trainer.evaluate()
    val_loss = metrics.get("eval_loss", float("inf"))
    trial.report(val_loss, step=args.max_epochs)
    if trial.should_prune():
        raise optuna.TrialPruned()
    # Clean GPU memory between trials.
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_loss


def main():
    args = parse_args()
    try:
        import optuna
    except ImportError:
        raise SystemExit("optuna is not installed; pip install -r requirements.txt")
    study = optuna.create_study(study_name=args.study_name, storage=args.storage,
                                load_if_exists=True, direction="minimize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda t: objective(t, args), n_trials=args.trials,
                   timeout=args.timeout)
    print(f"Best value: {study.best_value:.4f} params: {study.best_params}")
    best_path = os.path.join(args.output_root, "best_config.json")
    os.makedirs(args.output_root, exist_ok=True)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({"params": study.best_params, "value": study.best_value,
                   "study": args.study_name}, f, indent=2)
    print(f"Wrote {best_path}. Retrain with scripts/train_model.py using these values.")


if __name__ == "__main__":
    main()
