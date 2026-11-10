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

from open_fret.loss_training import (TabLossTrainer, add_loss_arguments,
                                     loss_config_from_args, validate_loss_cache)
from open_fret.losses import LossConfig
from open_fret.token_metadata import TokenMetadata

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
    add_loss_arguments(p)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--timeout", type=float, default=None,
                   help="Global study timeout in seconds.")
    p.add_argument("--time-limit", type=float, default=2700,
                   help="Per-trial wall-clock cap in seconds (backstop, default 45 min).")
    p.add_argument("--storage", default="sqlite:///hpo-dev.db")
    p.add_argument("--tokenizer-dir", default="dataset/processed/tab_tokenizer")
    p.add_argument("--cache-dir", default="dataset/processed/cached-dev-hpo-split-v1")
    p.add_argument("--output-root", default="models/hpo-dev-v1")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="If > 0, overrides --max-epochs with a step budget.")
    p.add_argument("--max-epochs", type=int, default=5,
                   help="Epoch bound per trial (binds first; time cap is backstop).")
    p.add_argument("--study-name", default="dev-hpo-v4")
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--gpu-monitor-interval", type=float, default=5.0,
                   help="Seconds between nvidia-smi utilization polls per trial "
                        "(0 disables GPU monitoring).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument('--fixture-space', action='store_true', help='Tiny CPU acceptance search, not production tuning')
    p.add_argument('--pruner', choices=['median', 'threshold', 'none'], default='median')
    p.add_argument('--pruner-warmup-steps', type=int, default=1000)
    p.add_argument('--prune-threshold', type=float, default=None)
    p.add_argument("--init-from", default=None,
                   help="Optional pretrained T5 endpoint to fine-tune per trial "
                        "instead of training from scratch.")
    p.add_argument("--sample-songs", type=int, default=5,
                   help="Fixed validation windows exported per completed trial "
                   "(0 disables trial sample export).")
    p.add_argument("--sample-data-dir", default=None,
                   help="Individual-window dir for trial samples "
                   "(required for export; skipped with notice if absent).")
    p.add_argument("--sample-val-files", default=None,
                   help="val_files.txt listing the sample panel source windows.")
    p.add_argument("--sample-beams", type=int, default=2,
                   help="Anchored beams for trial sample decoding (2 keeps "
                        "multi-trial studies fast; training eval uses 4).")
    return p.parse_args()


def suggest(trial, fixture=False):
    if fixture:
        return {'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True),
                'dropout': trial.suggest_float('dropout', 0, .1),
                'warmup_examples': 0, 'decay_after_examples': 10,
                'decay_factor': .1, 'batch_size': 1, 'grad_accum': 1}
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
        """Stop training after ``limit`` seconds (per-trial cap)."""
        def __init__(self, limit):
            self.limit = limit
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.t0 > self.limit:
                control.should_training_stop = True
            return control

    class LossHistoryCallback(_TC):
        """Collect train/eval loss curves; prune on intermediate eval loss."""
        def __init__(self, trial, history_path=None):
            self.trial = trial
            self.history_path = history_path
            self.train_curve: list = []
            self.eval_curve: list = []

        def persist(self):
            # Persist before pruning/failure too, not just at successful trial end.
            self.trial.set_user_attr("train_component_curve", self.train_curve)
            self.trial.set_user_attr("eval_component_curve", self.eval_curve)
            for prefix, curve, key in (("train", self.train_curve, "loss"),
                                       ("eval", self.eval_curve, "eval_loss")):
                self.trial.set_user_attr(prefix + "_loss_curve", [row[key] for row in curve])
                self.trial.set_user_attr(prefix + "_loss_steps", [row["step"] for row in curve])
            if self.history_path:
                with open(self.history_path, "w", encoding="utf-8") as f:
                    json.dump({"train": self.train_curve, "eval": self.eval_curve}, f, indent=2)

        @staticmethod
        def components(logs):
            return {key: float(value) for key, value in logs.items()
                    if isinstance(value, (int, float)) and (
                        "loss" in key or "smoothing" in key or "note_positions" in key)}

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.train_curve.append(
                    {"step": state.global_step, **self.components(logs)})
                self.persist()
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and "eval_loss" in metrics:
                self.eval_curve.append(
                    {"step": state.global_step,
                     **self.components(metrics)})
                self.persist()
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


class GPUMonitor:
    """Background nvidia-smi sampler: utilization + memory curve per trial.

    Polls ``nvidia-smi`` every ``interval`` seconds on a daemon thread.
    No new dependencies; disabled gracefully when no GPU/nvidia-smi exists.
    """

    def __init__(self, interval=5.0, gpu_index=0):
        import threading
        self.interval = interval
        self.gpu_index = gpu_index
        self.samples: list = []
        self._stop = threading.Event()
        self._thread = None
        self.enabled = interval > 0 and self._probe()

    def _probe(self):
        import shutil
        import subprocess
        if shutil.which("nvidia-smi") is None:
            return False
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", f"--id={self.gpu_index}"],
                capture_output=True, text=True, timeout=15, check=True)
            return True
        except Exception:
            return False

    def _poll_once(self):
        import subprocess
        import time
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", f"--id={self.gpu_index}"],
                capture_output=True, text=True, timeout=15, check=True).stdout
            util_s, mem_s = out.strip().split(",")
            self.samples.append({"t": round(time.time() - self.t0, 1),
                                 "util_pct": float(util_s.strip()),
                                 "mem_mb": float(mem_s.strip())})
        except Exception:
            pass

    def start(self):
        import threading
        import time
        if not self.enabled:
            return
        self.t0 = time.time()
        self._stop.clear()
        self._poll_once()

        def loop():
            while not self._stop.wait(self.interval):
                self._poll_once()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self.interval + 5)
            self._thread = None

    def summary(self):
        if not self.samples:
            return {"samples": [], "avg_util_pct": 0.0, "avg_mem_mb": 0.0,
                    "max_util_pct": 0.0, "max_mem_mb": 0.0, "n": 0}
        utils = [s["util_pct"] for s in self.samples]
        mems = [s["mem_mb"] for s in self.samples]
        return {"samples": self.samples,
                "avg_util_pct": round(sum(utils) / len(utils), 1),
                "avg_mem_mb": round(sum(mems) / len(mems), 1),
                "max_util_pct": max(utils), "max_mem_mb": max(mems),
                "n": len(self.samples)}


def objective(trial, args):
    import torch
    from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                              Adafactor, set_seed)
    from torch.optim.lr_scheduler import LambdaLR
    from datasets import load_from_disk
    from config import get_tiny_tab_config
    import optuna

    params = suggest(trial, getattr(args, 'fixture_space', False))
    loss_config = loss_config_from_args(args)
    validate_loss_cache(args.cache_dir, loss_config)
    trial.set_user_attr("loss_config", loss_config.to_dict())
    trial.set_user_attr('seed', args.seed + trial.number)
    trial.set_user_attr('budget', {'max_steps': args.max_steps, 'max_epochs': args.max_epochs,
                                   'seconds': args.time_limit, 'objective': 'validation loss',
                                   'pruner': getattr(args, 'pruner', 'median')})
    eff_batch = params["batch_size"] * params["grad_accum"]
    warmup_steps = max(0, round(params["warmup_examples"] / eff_batch))
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
    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, "trial_params.json"), "w", encoding="utf-8") as f:
        json.dump({"trial": trial.number, "params": params, "init": init_info,
                   "loss": loss_config.to_dict(),
                   "warmup_steps": warmup_steps, "decay_start_step": decay_start,
                   "effective_batch": eff_batch}, f, indent=2)

    optimizer = Adafactor(model.parameters(), lr=params["lr"],
                          relative_step=False, warmup_init=False)
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda s: burnin_decay_factor(
            s, warmup_steps, decay_start, params["decay_factor"]))

    history_cb = LossHistoryCallback(
        trial, os.path.join(out_dir, f"trial_{trial.number}_components.json"))
    targs = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["grad_accum"],
        num_train_epochs=args.max_epochs, max_steps=args.max_steps,
        learning_rate=params["lr"],
        weight_decay=0.0, lr_scheduler_type="constant",
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="no", logging_strategy="steps", logging_steps=50,
        fp16=torch.cuda.is_available(), report_to="none",
        seed=args.seed + trial.number)
    trainer = TabLossTrainer(model=model, args=targs, train_dataset=train_ds,
                       loss_config=loss_config, loss_tokenizer=tokenizer,
                      eval_dataset=val_ds, optimizers=(optimizer, scheduler),
                       callbacks=[TimeCapCallback(args.time_limit), history_cb])
    trial.set_user_attr("token_metadata", trainer.token_metadata.manifest())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gpu = GPUMonitor(interval=args.gpu_monitor_interval)
    gpu.start()
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
    finally:
        gpu.stop()
        gs = gpu.summary()
        trial.set_user_attr("gpu_util_curve", gs["samples"])
        trial.set_user_attr("avg_gpu_util_pct", gs["avg_util_pct"])
        trial.set_user_attr("avg_gpu_mem_mb", gs["avg_mem_mb"])
        trial.set_user_attr("max_gpu_util_pct", gs["max_util_pct"])
        trial.set_user_attr("max_gpu_mem_mb", gs["max_mem_mb"])
    metrics = trainer.evaluate()
    val_loss = float(metrics.get("eval_loss", float("inf")))
    trainer.save_model(os.path.join(out_dir, 'final'))
    tokenizer.save_pretrained(os.path.join(out_dir, 'final'))
    # Fixed-panel generated examples stay connected to the trial so weight
    # choices can be judged later by listening, not just by eval_loss.
    try:
        from export_samples import export_from_model
        from train_model import select_sample_songs
        sample_songs = []
        if args.sample_songs > 0 and args.sample_data_dir and args.sample_val_files:
            sample_songs = select_sample_songs(
                args.sample_val_files, args.sample_data_dir,
                k=args.sample_songs, tokenizer=tokenizer)
        if sample_songs:
            sample_manifest = export_from_model(
                tokenizer, model, sample_songs,
                os.path.join(out_dir, "samples"),
                title_prefix=f"trial {trial.number} — ",
                beams=args.sample_beams, decoding_mode="anchored")
            sample_manifest["trial"] = trial.number
            sample_manifest["loss"] = loss_config.to_dict()
            sample_manifest["val_loss"] = val_loss
            with open(os.path.join(out_dir, "samples", "manifest.json"),
                      "w", encoding="utf-8") as f:
                json.dump(sample_manifest, f, indent=2)
            trial.set_user_attr("sample_export", {
                "songs": len(sample_manifest["songs"]),
                "viewer": f"trial_{trial.number}/samples/viewer.html"
                if "viewer" in sample_manifest else None,
                "val_loss": val_loss})
        else:
            print("trial sample export skipped (need --sample-data-dir, "
                  "--sample-val-files and --sample-songs > 0)")
            trial.set_user_attr("sample_export", {"songs": 0, "viewer": None,
                                                  "val_loss": val_loss})
    except Exception as e:  # noqa: BLE001 - examples must never fail a trial
        print(f"trial sample export failed (non-fatal): {e}")
        trial.set_user_attr("sample_export", {"songs": 0, "viewer": None,
                                              "error": f"{type(e).__name__}: {e}",
                                              "val_loss": val_loss})

    history = {"trial": trial.number, "params": params,
               "loss": loss_config.to_dict(),
               "token_metadata": trainer.token_metadata.manifest(),
               "train_loss_curve": history_cb.train_curve,
               "eval_loss_curve": history_cb.eval_curve,
               "gpu_util_curve": trial.user_attrs.get("gpu_util_curve", []),
               "avg_gpu_util_pct": trial.user_attrs.get("avg_gpu_util_pct", 0.0),
               "avg_gpu_mem_mb": trial.user_attrs.get("avg_gpu_mem_mb", 0.0),
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


def run_study(args):
    loss_config = loss_config_from_args(args)
    validate_loss_cache(args.cache_dir, loss_config)
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
    if args.pruner == 'threshold':
        if args.prune_threshold is None:
            raise ValueError('--pruner threshold requires --prune-threshold')
        pruner = optuna.pruners.ThresholdPruner(upper=args.prune_threshold,
                                                n_warmup_steps=args.pruner_warmup_steps)
    elif args.pruner == 'none':
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=args.pruner_warmup_steps)
    study = optuna.create_study(study_name=args.study_name, storage=args.storage,
                                load_if_exists=True, direction='minimize', pruner=pruner,
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    stored_loss = study.user_attrs.get("loss_config")
    if stored_loss is None and study.trials:
        stored_loss = LossConfig().to_dict()
    if stored_loss is not None and stored_loss != loss_config.to_dict():
        raise ValueError("Study loss configuration differs; use a separate study name per loss ablation")
    study.set_user_attr("loss_config", loss_config.to_dict())
    metadata = TokenMetadata.from_tokenizer(tok, loss_config.max_fret).manifest()
    stored_metadata = study.user_attrs.get("token_metadata")
    if stored_metadata is not None and stored_metadata != metadata:
        raise ValueError("Study tokenizer/pitch metadata differs; use a new study name")
    study.set_user_attr("token_metadata", metadata)
    provenance = {'train_fingerprint': ds['train']._fingerprint,
                  'validation_fingerprint': ds['validation']._fingerprint,
                  'seed': args.seed, 'init_from': args.init_from, 'fixture_space': args.fixture_space}
    previous = study.user_attrs.get('data_protocol')
    if previous is not None and previous != provenance:
        raise ValueError('Study data/init/search protocol differs; use a new study name')
    study.set_user_attr('data_protocol', provenance)
    def guarded_objective(trial):
        import gc
        import torch
        try:
            return objective(trial, args)
        except Exception as e:
            trial.set_user_attr('failure', f'{type(e).__name__}: {e}')
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    study.optimize(guarded_objective, n_trials=args.trials, timeout=args.timeout,
                    catch=(RuntimeError,), gc_after_trial=True, n_jobs=1)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    os.makedirs(args.output_root, exist_ok=True)
    with open(os.path.join(args.output_root, 'study_summary.json'), 'w') as f:
        json.dump({'study': args.study_name, 'direction': 'minimize', 'data_protocol': provenance,
                   'requested_budget': vars(args), 'trials': [dict(number=t.number, state=t.state.name,
                    value=t.value, params=t.params, user_attrs=t.user_attrs) for t in study.trials]}, f, indent=2)
    if not completed:
        print('No completed trials; study_summary.json records failures/pruning, no winner exported')
        return
    print(f"Best value: {study.best_value:.4f} params: {study.best_params}")
    best_path = os.path.join(args.output_root, "best_config.json")
    os.makedirs(args.output_root, exist_ok=True)
    with open(best_path, "w", encoding="utf-8") as f:
        selected = study.best_trial.user_attrs['params_json']
        effective = selected['batch_size'] * selected['grad_accum']
        training_args = {key: selected[key] for key in ('lr', 'dropout', 'batch_size', 'grad_accum', 'decay_factor')}
        training_args.update(warmup_steps=max(0, round(selected['warmup_examples'] / effective)),
                             decay_start_step=max(max(0, round(selected['warmup_examples'] / effective)) + 1,
                                                  round(selected['decay_after_examples'] / effective)),
                             seed=args.seed, init_from=args.init_from)
        training_args.update(pitch_loss_alpha=loss_config.pitch_alpha,
                             equivalence_smoothing=loss_config.smoothing,
                             ce_loss_weight=loss_config.ce_weight, event_type_alpha=loss_config.type_alpha,
                             conditional_pitch=loss_config.conditional_pitch,
                             position_alpha=loss_config.position_alpha,
                             span_alpha=loss_config.span_alpha,
                             span_margin=loss_config.span_margin,
                             stretch_alpha=loss_config.stretch_alpha,
                             stretch_margin=loss_config.stretch_margin)
        json.dump({"params": selected, "training_args": training_args, "value": study.best_value,
                   "loss": loss_config.to_dict(),
                   "study": args.study_name, "init_from": args.init_from,
                   "user_attrs": study.best_trial.user_attrs}, f, indent=2)
    print(f"Wrote {best_path}. Retrain winner on full data with scripts/train_model.py.")
    # Saturation summary: one row per trial for the max-stable-batch analysis.
    summary_path = os.path.join(args.output_root, "batch_saturation.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("trial,state,batch_size,grad_accum,eff_batch,lr,"
                "val_loss,peak_mem_gb,avg_gpu_util_pct,avg_gpu_mem_mb,"
                "stable,loss_drop_per_step,"
                "train_samples_per_second,oom\n")
        for t in study.trials:
            a = t.user_attrs
            p = t.params
            eff = p.get("batch_size", 0) * p.get("grad_accum", 0)
            f.write(f"{t.number},{t.state.name},{p.get('batch_size')},"
                    f"{p.get('grad_accum')},{eff},{p.get('lr')},"
                    f"{t.value if t.value is not None else ''},"
                    f"{a.get('peak_mem_gb', '')},"
                    f"{a.get('avg_gpu_util_pct', '')},"
                    f"{a.get('avg_gpu_mem_mb', '')},"
                    f"{a.get('stable', '')},"
                    f"{a.get('loss_drop_per_step', '')},"
                    f"{a.get('train_samples_per_second', '')},"
                    f"{a.get('oom', '')}\n")
    print(f"Wrote {summary_path} (batch vs memory vs stability).")


def main():
    import fcntl
    import torch
    import tempfile
    args = parse_args()
    root = '/tmp/opencode' if os.path.isdir('/tmp/opencode') else tempfile.gettempdir()
    device = str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'cuda0')) if torch.cuda.is_available() else 'cpu'
    with open(os.path.join(root, f'open-fret-hpo-{os.getuid()}-{device}.lock'), 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another Open-Fret HPO process holds the {device} resource lock')
        run_study(args)


if __name__ == "__main__":
    main()
