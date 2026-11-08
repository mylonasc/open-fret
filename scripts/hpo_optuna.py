"""Optuna hyperparameter search (Issue #11).

Tune on the small dev cache (default: cached-dev-hpo-split-v1), validate on
its held-out val split, then retrain the winner on full data. Test data is
never used here.

Schedule per trial (all thresholds in *seen examples*, converted to steps
via the trial's effective batch size):
  burn-in  = linear warmup 0 -> lr over ``warmup_examples``
  hold     = constant lr until ``decay_after_examples``
  decay    = single multiplicative decay by ``decay_factor`` at the threshold

Each trial is capped at ``--time-limit`` seconds (default 30 min) via a
TrainerCallback. Train/val loss curves are stored per trial as
``trial_<n>_history.json`` and as trial user_attrs.

Example:
    python scripts/hpo_optuna.py --trials 12 --study-name dev-hpo-v1 \\
        --storage sqlite:///hpo-dev.db
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPACE = {
    "lr": (1e-5, 1e-3, True),
    "dropout": (0.0, 0.2),
    "warmup_examples": [10000, 50000, 100000],
    "decay_after_examples": [10000, 50000, 100000],
    "decay_factor": [0.1, 0.3, 0.5],
    # Sized to saturate 24GB VRAM on this tiny model (batch 32 ~= 4.6GB,
    # batch 96 ~= 10GB). Upper choices may OOM -> recorded as oom trials.
    "batch_size": [32, 64, 96, 128, 192, 256],
    "grad_accum": [1, 2],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--timeout", type=float, default=None,
                   help="Global study timeout in seconds.")
    p.add_argument("--time-limit", type=float, default=1800,
                   help="Per-trial wall-clock cap in seconds (default 30 min).")
    p.add_argument("--study-name", default="dev-hpo-v1")
    p.add_argument("--storage", default="sqlite:///hpo-dev.db")
    p.add_argument("--tokenizer-dir", default="dataset/processed/tab_tokenizer")
    p.add_argument("--cache-dir", default="dataset/processed/cached-dev-hpo-split-v1")
    p.add_argument("--output-root", default="models/hpo-dev-v1")
    p.add_argument("--max-steps", type=int, default=50000,
                   help="Safety step cap per trial; the time limit binds first.")
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-from", default=None,
                   help="Optional pretrained T5 endpoint to fine-tune per trial "
                        "instead of training from scratch.")
    return p.parse_args()


def suggest(trial):
    return {
        "lr": trial.suggest_float("lr", *SPACE["lr"][:2], log=SPACE["lr"][2]),
        "dropout": trial.suggest_float("dropout", *SPACE["dropout"]),
        "warmup_examples": trial.suggest_categorical(
            "warmup_examples", SPACE["warmup_examples"]),
        "decay_after_examples": trial.suggest_categorical(
            "decay_after_examples", SPACE["decay_after_examples"]),
        "decay_factor": trial.suggest_categorical(
            "decay_factor", SPACE["decay_factor"]),
        "batch_size": trial.suggest_categorical("batch_size", SPACE["batch_size"]),
        "grad_accum": trial.suggest_categorical("grad_accum", SPACE["grad_accum"]),
    }


# Bind the base class properly (keeps isinstance checks in Trainer working).
try:
    from transformers import TrainerCallback as _TC

    class TimeCapCallback(_TC):  # noqa: F811
        def __init__(self, limit):
            self.limit = limit
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.t0 > self.limit:
                control.should_training_stop = True
            return control

    class LossHistoryCallback(_TC):
        """Collect train/eval loss curves; prune on intermediate eval loss."""
        def __init__(self, trial):
            self.trial = trial
            self.train_curve: list = []
            self.eval_curve: list = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.train_curve.append(
                    {"step": state.global_step, "loss": float(logs["loss"])})
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and "eval_loss" in metrics:
                self.eval_curve.append(
                    {"step": state.global_step,
                     "eval_loss": float(metrics["eval_loss"])})
                self.trial.report(float(metrics["eval_loss"]),
                                  step=state.global_step)
                if self.trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
            return control
except ImportError:  # transformers not installed (pure-python test envs)
    TimeCapCallback = None
    LossHistoryCallback = None


def burnin_decay_factor(step, warmup_steps, decay_start_step, decay_factor):
    if step < warmup_steps:
        return float(step) / float(max(warmup_steps, 1))
    if step < decay_start_step:
        return 1.0
    return float(decay_factor)


def objective(trial, args):
    import torch
    from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                              Adafactor, set_seed)
    from torch.optim.lr_scheduler import LambdaLR
    from datasets import load_from_disk
    from config import get_tiny_tab_config
    import optuna

    params = suggest(trial)
    eff_batch = params["batch_size"] * params["grad_accum"]
    warmup_steps = max(1, round(params["warmup_examples"] / eff_batch))
    decay_start = max(warmup_steps + 1,
                      round(params["decay_after_examples"] / eff_batch))
    set_seed(args.seed + trial.number)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    from open_fret.training import load_init_model
    model, init_info = load_init_model(
        tokenizer, init_from=args.init_from,
        tiny_config_fn=get_tiny_tab_config, dropout=params["dropout"])
    print(f"Trial {trial.number} params={params} warmup_steps={warmup_steps} "
          f"decay_start_step={decay_start}", flush=True)
    dataset = load_from_disk(args.cache_dir)
    if "validation" not in dataset:
        raise ValueError("HPO requires a validation split in the cached dataset.")
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    out_dir = os.path.join(args.output_root, f"trial_{trial.number}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "trial_params.json"), "w", encoding="utf-8") as f:
        json.dump({"trial": trial.number, "params": params, "init": init_info,
                   "warmup_steps": warmup_steps, "decay_start_step": decay_start,
                   "effective_batch": eff_batch}, f, indent=2)

    optimizer = Adafactor(model.parameters(), lr=params["lr"],
                          relative_step=False, warmup_init=False)
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda s: burnin_decay_factor(
            s, warmup_steps, decay_start, params["decay_factor"]))

    history_cb = LossHistoryCallback(trial)
    targs = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["grad_accum"],
        max_steps=args.max_steps, learning_rate=params["lr"],
        weight_decay=0.0, lr_scheduler_type="constant",
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="no", logging_strategy="steps", logging_steps=50,
        fp16=torch.cuda.is_available(), report_to="none",
        seed=args.seed + trial.number)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, optimizers=(optimizer, scheduler),
                      callbacks=[TimeCapCallback(args.time_limit), history_cb])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        train_result = trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            trial.set_user_attr("oom", True)
            trial.set_user_attr("batch_size", params["batch_size"])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        raise
    metrics = trainer.evaluate()
    val_loss = float(metrics.get("eval_loss", float("inf")))

    history = {"trial": trial.number, "params": params,
               "train_loss_curve": history_cb.train_curve,
               "eval_loss_curve": history_cb.eval_curve,
               "final_eval": metrics, "val_loss": val_loss}
    with open(os.path.join(out_dir, f"trial_{trial.number}_history.json"),
              "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    trial.set_user_attr("train_loss_curve",
                        [p["loss"] for p in history_cb.train_curve])
    trial.set_user_attr("train_loss_steps",
                        [p["step"] for p in history_cb.train_curve])
    trial.set_user_attr("eval_loss_curve",
                        [p["eval_loss"] for p in history_cb.eval_curve])
    trial.set_user_attr("eval_loss_steps",
                        [p["step"] for p in history_cb.eval_curve])
    trial.set_user_attr("params_json", params)
    # Memory saturation + stability: the raw material for "max stable batch".
    peak_gb = (float(torch.cuda.max_memory_allocated() / 1e9)
               if torch.cuda.is_available() else 0.0)
    trial.set_user_attr("peak_mem_gb", peak_gb)
    trial.set_user_attr("oom", False)
    first_eval = (history_cb.eval_curve[0]["eval_loss"]
                  if history_cb.eval_curve else float("inf"))
    best_eval = min([p["eval_loss"] for p in history_cb.eval_curve],
                    default=float("inf"))
    best_step = next((p["step"] for p in history_cb.eval_curve
                      if p["eval_loss"] == best_eval), 0)
    stable = (best_eval < float("inf") and best_eval < first_eval
              and best_eval == best_eval)  # NaN-safe
    trial.set_user_attr("stable", bool(stable))
    trial.set_user_attr("loss_drop_per_step",
                        float((first_eval - best_eval) / max(best_step, 1))
                        if stable else 0.0)
    train_sps = (train_result.metrics.get("train_samples_per_second", None)
                 if train_result is not None and train_result.metrics else None)
    if train_sps is not None:
        trial.set_user_attr("train_samples_per_second", float(train_sps))

    del trainer, model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_loss


def main():
    args = parse_args()
    try:
        import optuna
    except ImportError:
        raise SystemExit("optuna is not installed; pip install -r requirements.txt")
    try:
        from transformers import AutoTokenizer
        from datasets import load_from_disk
        tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        ds = load_from_disk(args.cache_dir)
        print(f"HPO data: cache={args.cache_dir} "
              f"train={len(ds['train'])} val={len(ds['validation'])} "
              f"vocab={len(tok)}", flush=True)
    except Exception as e:
        raise SystemExit(f"HPO data check failed: {e}")
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
                   "study": args.study_name, "init_from": args.init_from,
                   "user_attrs": study.best_trial.user_attrs}, f, indent=2)
    print(f"Wrote {best_path}. Retrain winner on full data with scripts/train_model.py.")
    # Saturation summary: one row per trial for the max-stable-batch analysis.
    summary_path = os.path.join(args.output_root, "batch_saturation.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("trial,state,batch_size,grad_accum,eff_batch,lr,"
                "val_loss,peak_mem_gb,stable,loss_drop_per_step,"
                "train_samples_per_second,oom\n")
        for t in study.trials:
            a = t.user_attrs
            p = t.params
            eff = p.get("batch_size", 0) * p.get("grad_accum", 0)
            f.write(f"{t.number},{t.state.name},{p.get('batch_size')},"
                    f"{p.get('grad_accum')},{eff},{p.get('lr')},"
                    f"{t.value if t.value is not None else ''},"
                    f"{a.get('peak_mem_gb', '')},{a.get('stable', '')},"
                    f"{a.get('loss_drop_per_step', '')},"
                    f"{a.get('train_samples_per_second', '')},"
                    f"{a.get('oom', '')}\n")
    print(f"Wrote {summary_path} (batch vs memory vs stability).")


if __name__ == "__main__":
    main()
