"""Shared opt-in Trainer integration for normal training and Optuna."""

from __future__ import annotations

import json
import os

from transformers import Trainer

from .alphatex import PARSER_VERSION, SUPPORTED_PARSER_VERSIONS
from .losses import LossConfig, TabLoss
from .token_metadata import TokenMetadata


def add_loss_arguments(parser):
    parser.add_argument("--ce-loss-weight", type=float, default=1.0,
                        help="Weight of (optionally equivalence-smoothed) CE.")
    parser.add_argument("--pitch-loss-alpha", type=float, default=0.0,
                       help="Opt-in CE + alpha * same-pitch marginal loss (e.g. 0.3).")
    parser.add_argument("--equivalence-smoothing", type=float, default=0.0,
                       help="Opt-in probability mass for same-pitch plain TAB alternatives (e.g. 0.1).")
    parser.add_argument("--loss-max-fret", type=int, default=24,
                        help="Maximum valid fret for domain-aware supervision.")


def loss_config_from_args(args):
    return LossConfig(ce_weight=args.ce_loss_weight, pitch_alpha=args.pitch_loss_alpha,
                      smoothing=args.equivalence_smoothing,
                      max_fret=args.loss_max_fret)


def validate_loss_cache(cache_dir, config):
    """Do not start new loss experiments on known obsolete timing supervision."""
    if not config.enabled:
        return
    path = os.path.join(cache_dir, "cache_manifest.json")
    try:
        with open(path, encoding="utf-8") as f:
            version = json.load(f).get("parser_version")
    except FileNotFoundError:
        version = None
    if version not in SUPPORTED_PARSER_VERSIONS:
        raise ValueError(
            f"Cache {cache_dir!r} has obsolete/unversioned AlphaTex supervision. "
            "Reprocess into a NEW directory with scripts/01_process.py, split, "
            "and rebuild tokenizer/cache before domain-aware loss experiments; "
            f"expected parser_version={PARSER_VERSION!r}.")


class TabLossTrainer(Trainer):
    """CE remains the default; custom losses use microbatch means.

    Resume refuses changed loss settings/vocabulary semantics. Fine-tuning with
    a different objective is still possible via --init-from into a new run.
    """

    def __init__(self, *args, loss_config=None, loss_tokenizer=None, **kwargs):
        self.tab_loss_config = loss_config or LossConfig()
        self.token_metadata = TokenMetadata.from_tokenizer(
            loss_tokenizer, self.tab_loss_config.max_fret)
        self._loss_sums = {"train": {}, "eval": {}}
        super().__init__(*args, **kwargs)
        if self.tab_loss_config.enabled and self.label_smoother is not None:
            raise ValueError("Do not combine HF label smoothing with equivalence supervision")
        self.tab_objective = (TabLoss(self.token_metadata, self.tab_loss_config).to(self.args.device)
                              if self.tab_loss_config.enabled else None)
        self.model.config.open_fret_loss = self.tab_loss_config.to_dict()
        self.model.config.open_fret_token_metadata = self.token_metadata.manifest()
        if self.tab_loss_config.enabled:
            # HF versions differ in gradient accumulation plumbing. This loss
            # is a mean per microbatch, NOT normalized over the accumulated batch.
            self.model_accepts_loss_kwargs = False
            self.loss_is_scaled_for_ga = False

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        with open(os.path.join(resume_from_checkpoint, "config.json"), encoding="utf-8") as f:
            saved = json.load(f)
        saved_loss = dict(saved.get("open_fret_loss", LossConfig().to_dict()))
        saved_loss.setdefault("ce_weight", 1.0)
        saved_loss["version"] = 2
        if saved_loss != self.tab_loss_config.to_dict():
            raise ValueError("Checkpoint loss differs from requested loss; use identical flags for resume, "
                             "or --init-from for a new experiment")
        if self.tab_loss_config.enabled and saved.get("open_fret_token_metadata") != self.token_metadata.manifest():
            raise ValueError("Checkpoint tokenizer/pitch metadata differs from this run")
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if self.tab_loss_config.enabled:
            self._signature_columns = list(dict.fromkeys(
                [*self._signature_columns, "loss_tuning", "loss_capo"]))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self.tab_loss_config.enabled:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items()
                        if k not in ("labels", "loss_tuning", "loss_capo")}
        if "decoder_input_ids" not in model_inputs:
            unwrapped = self.accelerator.unwrap_model(model)
            model_inputs["decoder_input_ids"] = unwrapped.prepare_decoder_input_ids_from_labels(labels)
        outputs = model(**model_inputs)
        self.tab_objective.to(outputs.logits.device)
        loss, parts = self.tab_objective(outputs.logits, labels, inputs["input_ids"],
                                        loss_tuning=inputs.get("loss_tuning"),
                                        loss_capo=inputs.get("loss_capo"))
        outputs["loss"] = loss
        bucket = self._loss_sums["train" if model.training else "eval"]
        terms = {"ce": parts["ce"] * parts["tokens"],
                 "pitch": parts["pitch"] * parts["notes"],
                 "smoothing_delta": parts["smoothing_delta"] * parts["tokens"],
                 "tokens": parts["tokens"], "notes": parts["notes"],
                 "smoothed_notes": parts["smoothed_notes"]}
        for name, value in terms.items():
            bucket[name] = bucket.get(name, 0) + value
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        phase = "eval" if any(k.startswith("eval_") for k in logs) else "train"
        bucket = self._loss_sums[phase]
        if bucket:
            prefix = "eval_" if phase == "eval" else ""
            values = {k: float(v.item()) for k, v in bucket.items()}
            logs.update({
                prefix + "ce_loss": values["ce"] / max(values["tokens"], 1),
                prefix + "pitch_loss": values["pitch"] / max(values["notes"], 1),
                prefix + "equivalence_smoothing_delta": values["smoothing_delta"] / max(values["tokens"], 1),
                prefix + "note_positions": values["notes"],
                prefix + "smoothed_note_positions": values["smoothed_notes"],
                prefix + "pitch_loss_alpha": self.tab_loss_config.pitch_alpha,
                prefix + "equivalence_smoothing": self.tab_loss_config.smoothing,
                prefix + "ce_loss_weight": self.tab_loss_config.ce_weight,
                prefix + "weighted_ce_loss": self.tab_loss_config.ce_weight * values["ce"] / max(values["tokens"], 1),
                prefix + "weighted_pitch_loss": self.tab_loss_config.pitch_alpha * values["pitch"] / max(values["notes"], 1),
                prefix + "weighted_smoothing_delta": self.tab_loss_config.ce_weight * values["smoothing_delta"] / max(values["tokens"], 1),
            })
            bucket.clear()
        return super().log(logs, start_time=start_time)
