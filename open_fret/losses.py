"""Differentiable pitch marginal and token-local equivalence smoothing.

Neither loss decodes/argmaxes predictions. TS/EOS/techniques retain exact CE
supervision. Tuning is concert MIDI without capo; capo is added exactly once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .token_metadata import TokenMetadata, MAX_STRINGS


@dataclass(frozen=True)
class LossConfig:
    ce_weight: float = 1.0
    pitch_alpha: float = 0.0
    smoothing: float = 0.0
    max_fret: int = 24

    def __post_init__(self):
        if not math.isfinite(self.ce_weight) or self.ce_weight <= 0:
            raise ValueError("ce_weight must be finite and positive")
        if not math.isfinite(self.pitch_alpha) or self.pitch_alpha < 0:
            raise ValueError("pitch_alpha must be finite and nonnegative")
        if not math.isfinite(self.smoothing) or not 0 <= self.smoothing < 1:
            raise ValueError("smoothing must be finite and in [0, 1)")
        if not 0 <= self.max_fret <= 127:
            raise ValueError("max_fret must be between 0 and 127")

    @property
    def enabled(self):
        return bool(self.ce_weight != 1.0 or self.pitch_alpha or self.smoothing)

    def to_dict(self):
        return {"version": 2, **asdict(self)}


class TabLoss(nn.Module):
    """Compact lookup tables and sparse gathers from shared log probabilities.

    Candidate storage is strings × frets × technique variants, never vocab².
    Smoothing is limited to technique-free TAB targets and alternatives.
    """

    def __init__(self, metadata: TokenMetadata, config: LossConfig):
        super().__init__()
        if metadata.max_fret != config.max_fret:
            raise ValueError("Loss configuration and metadata fret limits differ")
        self.config = config
        if not config.enabled:
            # Native CE must not depend on interpreting conditioning tokens.
            self.register_buffer("plain", torch.zeros(len(metadata.tokens), dtype=torch.bool),
                                 persistent=False)
            return
        tunings = [metadata.tuning(t) for t in metadata.tokens]
        strings = MAX_STRINGS
        if any(len(t) > strings for t in tunings if t):
            raise ValueError(f"At most {strings} tuning strings are supported")
        vocab_size = len(metadata.tokens)
        tuning_table = torch.full((vocab_size, strings), -1, dtype=torch.long)
        capo_table = torch.full((vocab_size,), -1, dtype=torch.long)
        note_string = torch.full((vocab_size,), -1, dtype=torch.long)
        note_fret = torch.zeros(vocab_size, dtype=torch.long)
        plain = torch.zeros(vocab_size, dtype=torch.bool)
        is_tab = torch.tensor([t.startswith("TAB_") for t in metadata.tokens])
        candidates = {}
        for i, token in enumerate(metadata.tokens):
            if tunings[i] is not None:
                tuning_table[i, :len(tunings[i])] = torch.tensor(tunings[i])
            capo = metadata.capo(token)
            if capo is not None:
                capo_table[i] = capo
            note = metadata.note(i)
            if note is None:
                continue
            string, fret, technique = note
            if not 1 <= string <= strings or fret > config.max_fret:
                continue
            note_string[i], note_fret[i], plain[i] = string - 1, fret, technique is None
            candidates.setdefault((string - 1, fret), []).append(i)
        variants = max(map(len, candidates.values()), default=1)
        candidate_table = torch.full((strings, config.max_fret + 1, variants), -1,
                                    dtype=torch.long)
        plain_table = torch.full((strings, config.max_fret + 1), -1, dtype=torch.long)
        for (string, fret), ids in candidates.items():
            candidate_table[string, fret, :len(ids)] = torch.tensor(ids)
            for i in ids:
                if plain[i]:
                    plain_table[string, fret] = i
        for name, table in {
            "tunings": tuning_table, "capos": capo_table,
            "note_string": note_string, "note_fret": note_fret,
            "plain": plain, "is_tab": is_tab,
            "candidates": candidate_table, "plain_candidates": plain_table,
        }.items():
            self.register_buffer(name, table, persistent=False)

    def forward(self, logits, labels, input_ids, loss_tuning=None, loss_capo=None):
        if logits.shape[:-1] != labels.shape or logits.shape[-1] != len(self.plain):
            raise ValueError("Logits, labels and tokenizer vocabulary must agree")
        active = labels.ne(-100)
        if ((labels[active] < 0) | (labels[active] >= logits.shape[-1])).any():
            raise ValueError("Label outside tokenizer vocabulary")
        count = active.sum()
        zero = logits.reshape(-1)[:0].sum().float()
        if self.config.enabled and count:
            # Reuse CE's log-softmax for the auxiliary term: no second full
            # normalization and no [number_of_notes, vocabulary] score copy.
            log_probs = logits.float().log_softmax(-1)
            ce_loss = F.nll_loss(log_probs.flatten(0, 1), labels.flatten(), ignore_index=-100)
        else:
            ce_loss = (F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(),
                                       ignore_index=-100) if count else zero)
        metrics = {"ce": ce_loss.detach(), "pitch": zero.detach(),
                   "smoothing_delta": zero.detach(), "tokens": count.detach(),
                   "notes": zero.detach(), "smoothed_notes": zero.detach()}
        if not self.config.enabled:
            return ce_loss, metrics
        safe_labels = labels.masked_fill(~active, 0)
        note_mask = active & self.note_string[safe_labels].ge(0)
        if (active & self.is_tab[safe_labels] & ~note_mask).any():
            raise ValueError("Invalid TAB supervision for configured string/fret limits")
        if input_ids.ndim != 2 or input_ids.shape[0] != labels.shape[0] or input_ids.shape[1] < 2:
            raise ValueError("Pitch supervision requires CAPO and TUNING input prefixes")
        prefix = input_ids[:, :2]
        if ((prefix < 0) | (prefix >= len(self.plain))).any():
            raise ValueError("Input prefix outside tokenizer vocabulary")
        capo = self.capos[prefix[:, 0]]
        tuning = self.tunings[prefix[:, 1]]
        if loss_tuning is not None or loss_capo is not None:
            if loss_tuning is None or loss_capo is None:
                raise ValueError("Source conditioning requires both tuning and capo")
            if loss_tuning.shape != tuning.shape or loss_capo.shape != capo.shape:
                raise ValueError("Source conditioning has incorrect batch/string shape")
            if ((loss_tuning < -1) | (loss_tuning > 127)).any() or ((loss_capo < 0) | (loss_capo > 24)).any():
                raise ValueError("Source conditioning outside allowed MIDI/capo range")
            if ((capo >= 0) & capo.ne(loss_capo)).any():
                raise ValueError("Source capo disagrees with input prefix")
            known_tuning = tuning.ge(0).any(-1)
            if (known_tuning[:, None] & tuning.ne(loss_tuning)).any():
                raise ValueError("Source tuning disagrees with input prefix")
            tuning, capo = loss_tuning, loss_capo
        if (capo < 0).any() or tuning.lt(0).all(-1).any():
            raise ValueError("Expected CAPO_<n> then TUNING_<concert-MIDI...>; no standard-tuning fallback")
        positions = note_mask.nonzero()
        metrics["notes"] = note_mask.sum().detach()
        if not len(positions):
            return self.config.ce_weight * ce_loss, metrics
        pitch_sum, smooth_sum, smoothed_count = zero, zero, zero.detach()
        batch, time = positions.unbind(dim=1)
        target = labels[batch, time]
        target_string = self.note_string[target]
        open_pitch = tuning[batch, target_string]
        if open_pitch.lt(0).any():
            raise ValueError("Target string outside example's tuning")
        target_pitch = open_pitch + capo[batch] + self.note_fret[target]
        if ((target_pitch < 0) | (target_pitch > 127)).any():
            raise ValueError("Target pitch outside MIDI range")
        frets = target_pitch[:, None] - tuning[batch] - capo[batch, None]
        valid = tuning[batch].ge(0) & frets.ge(0) & frets.le(self.config.max_fret)
        frets = frets.clamp(0, self.config.max_fret)
        string_ids = torch.arange(tuning.shape[1], device=logits.device)
        if self.config.pitch_alpha:
            ids = self.candidates[string_ids, frets].flatten(1)
            mask = valid[:, :, None].expand(-1, -1, self.candidates.shape[-1]).flatten(1) & ids.ge(0)
            equivalent = log_probs[batch[:, None], time[:, None], ids.clamp_min(0)]
            equivalent = equivalent.masked_fill(~mask, -torch.inf)
            pitch_sum = -equivalent.logsumexp(-1).sum()
        if self.config.smoothing:
            ids = self.plain_candidates[string_ids, frets]
            mask = valid & ids.ge(0) & ids.ne(target[:, None]) & self.plain[target, None]
            alternatives = mask.sum(-1)
            eligible = alternatives.gt(0)
            alternatives_logp = log_probs[batch[:, None], time[:, None], ids.clamp_min(0)]
            alt_mean = (alternatives_logp.masked_fill(~mask, 0).sum(-1)
                        / alternatives.clamp_min(1))
            delta = log_probs[batch, time, target] - alt_mean
            smooth_sum = delta.masked_fill(~eligible, 0).sum()
            smoothed_count = eligible.sum()
        pitch_mean = pitch_sum / metrics["notes"].clamp_min(1)
        smooth_delta = self.config.smoothing * smooth_sum / count.clamp_min(1)
        loss = self.config.ce_weight * (ce_loss + smooth_delta) + self.config.pitch_alpha * pitch_mean
        metrics.update(pitch=pitch_mean.detach(), smoothing_delta=smooth_delta.detach(),
                       smoothed_notes=smoothed_count.detach())
        return loss, metrics
