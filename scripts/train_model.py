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


def burnin_decay_factor(step, warmup_steps, decay_start_step, decay_factor):
    """Linear burn-in -> hold -> single multiplicative decay (HPO schedule).

    Mirrors ``scripts/hpo_optuna.py`` so the full run uses the winning
    schedule exactly: warmup 0 -> lr over ``warmup_steps``, hold until
    ``decay_start_step``, then scale by ``decay_factor``.
    """
    if step < warmup_steps:
        return float(step) / float(max(warmup_steps, 1))
    if step < decay_start_step:
        return 1.0
    return float(decay_factor)


def select_sample_songs(val_files_path, data_dir, k=5):
    """Pick K fixed validation windows from distinct songs for sample export.

    ``val_files.txt`` entries look like ``<song>.tex__t<track>__c<capo>__w<win>.json``;
    grouping by the song prefix gives musically distinct samples (not five
    windows of the same song). Returns list of
    ``{name, midi_tokens, tab_tokens}`` dicts.
    """
    with open(val_files_path, encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if not names:
        raise ValueError(f"Empty split file: {val_files_path}")
    seen, picked = set(), []
    for name in names:
        song = name.split("__t")[0]
        if song in seen:
            continue
        seen.add(song)
        picked.append(name)
        if len(picked) >= k:
            break
    if len(picked) < k:  # tiny splits: pad with the first files in order
        for name in names:
            if name not in picked:
                picked.append(name)
            if len(picked) >= k:
                break
    samples = []
    for name in picked[:k]:
        path = os.path.join(data_dir, name)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        midi_tokens = data.get("midi") or data.get("input") or []
        tab_tokens = data.get("tab") or data.get("target") or []
        if not midi_tokens or not tab_tokens:
            raise ValueError(f"Sample file {name} has empty midi/tab streams.")
        samples.append({"name": os.path.splitext(name)[0],
                        "midi_tokens": list(midi_tokens),
                        "tab_tokens": list(tab_tokens)})
    return samples


class TabSampleCallback(TrainerCallback):
    """On every validation, decode the fixed sample songs with current weights.

    Uses the in-memory model (= the latest state at this step, equivalent to
    the newest checkpoint once it is written). Decoding is beam search with a
    no-repeat-ngram guard: pure greedy decoding of this small model collapses
    into 1-2 token loops that never emit EOS (81% loop rate on 100 val
    windows), which misrepresents what the model learned. Each eval writes::

        <output_dir>/samples/step-<global_step>/rules-<song>.{pred,ref,input}.txt
        <song>.pred.tex / <song>.pred.ascii + manifest.json

    Failures are logged, never fatal to training.
    """

    def __init__(self, tokenizer, samples, output_dir, run_name,
                 max_input_tokens=512, max_new_tokens=512, log_path=None,
                 beams=4, no_repeat_ngram=3):
        self.tokenizer = tokenizer
        self.samples = samples
        self.output_dir = output_dir
        self.run_name = run_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.log_path = log_path
        self.beams = max(beams, 1)
        self.no_repeat_ngram = no_repeat_ngram or None
        self.tokenizer = tokenizer
        self.samples = samples
        self.output_dir = output_dir
        self.run_name = run_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.log_path = log_path

    def _log(self, msg):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(msg + "\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs.get("model", None)
        if model is None or not self.samples:
            return control
        from open_fret.render import TabRenderer
        step = state.global_step
        epoch = round(state.epoch or 0, 3)
        out_dir = os.path.join(
            self.output_dir, "samples", f"step-{step:08d}")
        os.makedirs(out_dir, exist_ok=True)
        was_training = model.training
        model.eval()
        renderer = TabRenderer()
        manifest = {"run": self.run_name, "step": step, "epoch": epoch,
                    "eval_metrics": metrics or {}, "songs": []}
        try:
            import torch
            for s in self.samples:
                try:
                    enc = self.tokenizer(
                        " ".join(s["midi_tokens"]),
                        return_tensors="pt", truncation=True,
                        max_length=self.max_input_tokens)
                    enc = {k: v.to(model.device) for k, v in enc.items()}
                    with torch.no_grad():
                        out = model.generate(
                            **enc, max_new_tokens=self.max_new_tokens,
                            do_sample=False, num_beams=self.beams,
                            no_repeat_ngram_size=self.no_repeat_ngram,
                            early_stopping=True)
                    pred = self.tokenizer.decode(
                        out[0], skip_special_tokens=True).split()
                    base = "".join(
                        c if (c.isalnum() or c in "-_.") else "_"
                        for c in s["name"])[:120]
                    with open(os.path.join(out_dir, f"{base}.pred.txt"),
                              "w", encoding="utf-8") as f:
                        f.write(" ".join(pred))
                    with open(os.path.join(out_dir, f"{base}.ref.txt"),
                              "w", encoding="utf-8") as f:
                        f.write(" ".join(s["tab_tokens"]))
                    with open(os.path.join(out_dir, f"{base}.input.txt"),
                              "w", encoding="utf-8") as f:
                        f.write(" ".join(s["midi_tokens"]))
                    with open(os.path.join(out_dir, f"{base}.pred.tex"),
                              "w", encoding="utf-8") as f:
                        f.write(renderer.to_alphatex(pred, title=base))
                    with open(os.path.join(out_dir, f"{base}.pred.ascii"),
                              "w", encoding="utf-8") as f:
                        f.write(renderer.to_ascii(pred))
                    manifest["songs"].append(
                        {"name": s["name"], "file": base,
                         "n_pred": len(pred),
                         "n_ref": len(s["tab_tokens"])})
                except Exception as e:  # noqa: BLE001 - per-song isolation
                    self._log(f"Sample export failed for {s.get('name')}: {e}")
                    manifest["songs"].append(
                        {"name": s.get("name"), "error": str(e)})
        finally:
            if was_training:
                model.train()
        with open(os.path.join(out_dir, "manifest.json"),
                  "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        self._log(f"Step {step:06d} | exported {len(manifest['songs'])} "
                  f"tab samples -> {out_dir}")
        return control


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
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="Linear burn-in steps (HPO winner: 39 = 10k examples / 256 eff batch).")
    p.add_argument("--decay-start-step", type=int, default=None,
                   help="If set, hold LR until this step then multiply by "
                        "--decay-factor (HPO winner: 391). Enables the HPO "
                        "burn-in/decay schedule instead of cosine.")
    p.add_argument("--decay-factor", type=float, default=0.1,
                   help="Multiplicative LR decay applied at --decay-start-step.")
    p.add_argument("--sample-songs", type=int, default=5,
                   help="Fixed validation songs re-decoded to tab on every eval "
                        "(0 disables sample export).")
    p.add_argument("--sample-max-new-tokens", type=int, default=512)
    p.add_argument("--sample-beams", type=int, default=4,
                   help="Beam width for per-eval tab sample decoding "
                        "(greedy=1 loops degenerately on this model).")
    p.add_argument("--sample-no-repeat-ngram", type=int, default=3,
                   help="No-repeat-ngram guard for sample decoding (0 disables).")
    p.add_argument("--val-files", default="dataset/processed/val_files.txt",
                   help="Split list used to pick the fixed sample songs.")
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
    use_hpo_schedule = args.decay_start_step is not None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="constant" if use_hpo_schedule else "cosine",
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

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=10),
        TrainLoggerCallback(args.log_file),
    ]
    if args.sample_songs and int(args.sample_songs) > 0:
        try:
            samples = select_sample_songs(
                args.val_files, args.data_dir, k=int(args.sample_songs))
            print(f"Sample export: {len(samples)} fixed songs -> "
                  f"{output_dir}/samples/step-<step>/ "
                  f"({', '.join(s['name'][:40] for s in samples)})")
            callbacks.append(TabSampleCallback(
                tokenizer, samples, output_dir,
                run_name=os.path.basename(os.path.normpath(output_dir)),
                max_input_tokens=args.max_length,
                max_new_tokens=args.sample_max_new_tokens,
                log_path=args.log_file,
                beams=args.sample_beams,
                no_repeat_ngram=args.sample_no_repeat_ngram))
        except Exception as e:  # noqa: BLE001 - sampling never blocks training
            print(f"WARNING: sample export disabled ({e})")

    optimizers = (None, None)
    if use_hpo_schedule:
        # Exact HPO schedule: Adafactor without relative-step/warmup-init,
        # scaled by the burn-in/decay LambdaLR.
        from transformers import Adafactor
        from torch.optim.lr_scheduler import LambdaLR
        optimizer = Adafactor(model.parameters(), lr=args.lr,
                              relative_step=False, warmup_init=False)
        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda s: burnin_decay_factor(
                s, args.warmup_steps, args.decay_start_step,
                args.decay_factor))
        optimizers = (optimizer, scheduler)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=optimizers if use_hpo_schedule else None,
        callbacks=callbacks,
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
            "lr": args.lr, "warmup_steps": args.warmup_steps,
            "decay_start_step": args.decay_start_step,
            "decay_factor": args.decay_factor,
            "schedule": ("burnin-decay" if use_hpo_schedule else "cosine"),
            "dropout": args.dropout,
            "save_steps": args.save_steps, "eval_steps": args.eval_steps,
            "sample_songs": args.sample_songs,
            "sample_beams": args.sample_beams,
            "sample_no_repeat_ngram": args.sample_no_repeat_ngram,
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
