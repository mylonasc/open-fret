"""Shared generation and MIDI-anchored fingering beam search (issue #17).

Anchored mode copies attack multiplicity/time from input, never from a target.
It enforces pitch and same-onset string feasibility. Sustained occupancy across
different attacks is NOT enforced by this first slice.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

from .token_metadata import TokenMetadata


class UnsupportedInputError(ValueError):
    pass


class UnplayableInputError(UnsupportedInputError):
    pass


@lru_cache(maxsize=16384)
def _can_assign(domains, used=0):
    """Existence of a distinct-string matching; choose tightest domain first."""
    if not domains:
        return True
    available = tuple(mask & ~used for mask in domains)
    if any(mask == 0 for mask in available):
        return False
    union = 0
    for mask in available:
        union |= mask
    if union.bit_count() < len(available):
        return False
    index = min(range(len(available)), key=lambda i: available[i].bit_count())
    remaining = domains[:index] + domains[index + 1:]
    choices = available[index]
    while choices:
        bit = choices & -choices
        if _can_assign(remaining, used | bit):
            return True
        choices ^= bit
    return False


@dataclass(frozen=True)
class Slot:
    kind: str
    tick: int
    ids: tuple[int, ...]
    pitch: int | None = None


class AnchoredTemplate:
    def __init__(self, tokenizer, tokens, max_fret=24):
        self.vocab = tokenizer.get_vocab()
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise UnsupportedInputError("Tokenizer must define EOS")
        self.tokens = list(tokens)
        if self.tokens and self.tokens[-1] == tokenizer.eos_token:
            self.tokens.pop()
        if len(self.tokens) < 2:
            raise UnsupportedInputError("CAPO and TUNING prefixes required")
        self.capo = TokenMetadata.capo(self.tokens[0])
        self.tuning = TokenMetadata.tuning(self.tokens[1])
        if self.capo is None or self.tuning is None:
            raise UnsupportedInputError("Expected CAPO then concert-MIDI TUNING prefix")
        if not 0 <= max_fret <= 127:
            raise UnsupportedInputError("max_fret must be between 0 and 127")
        self.note_strings = {}
        candidates = {}
        metadata = TokenMetadata.from_tokenizer(tokenizer, max_fret)
        for token_id in range(len(metadata.tokens)):
            note = metadata.note(token_id)
            if note is None:
                continue
            pitch = metadata.pitch(token_id, self.tuning, self.capo)
            if pitch is not None:
                candidates.setdefault(pitch, []).append(token_id)
                self.note_strings[token_id] = note[0]
        slots, groups, tick = [], {}, 0
        for token in self.tokens[2:]:
            try:
                if token.startswith("TS_"):
                    delta = int(token[3:])
                    if delta < 0 or token not in self.vocab:
                        raise UnsupportedInputError(f"Unsupported time-shift token: {token}")
                    tick += delta
                    slots.append(Slot("time", tick, (self.vocab[token],)))
                elif token.startswith(("NO_", "NF_")):
                    pitch = int(token[3:])
                    if not 0 <= pitch <= 127:
                        raise UnsupportedInputError(f"Pitch outside MIDI range: {pitch}")
                    if token.startswith("NO_"):
                        ids = tuple(candidates.get(pitch, ()))
                        if not ids:
                            raise UnplayableInputError(f"No vocabulary fingering for pitch {pitch} at tick {tick}")
                        groups.setdefault(tick, []).append(len(slots))
                        slots.append(Slot("note", tick, ids, pitch))
                else:
                    raise UnsupportedInputError(f"Unsupported MIDI token: {token}")
            except (TypeError, ValueError) as e:
                if isinstance(e, UnsupportedInputError):
                    raise
                raise UnsupportedInputError(f"Malformed MIDI token: {token}") from e
        self.groups = groups
        self.slots = tuple(slots) + (Slot("eos", tick, (self.eos_id,)),)
        self.note_domains = {}
        for indices in groups.values():
            masks = tuple(sum(1 << (s - 1) for s in {self.note_strings[i] for i in self.slots[index].ids})
                          for index in indices)
            if not _can_assign(masks):
                raise UnplayableInputError(f"No distinct-string assignment for chord at tick {self.slots[indices[0]].tick}")
            for position, index in enumerate(indices):
                self.note_domains[index] = (indices[:position], masks[position + 1:])

    def allowed(self, position, prefix):
        slot = self.slots[position]
        if slot.kind != "note":
            return slot.ids
        previous, remaining = self.note_domains[position]
        used = 0
        for index in previous:
            selected = prefix[index + 1]  # decoder-start token is not a slot
            if selected not in self.slots[index].ids:
                raise RuntimeError("Decoder prefix violated the anchored template")
            used |= 1 << (self.note_strings[selected] - 1)
        return tuple(i for i in slot.ids
                     if not used & (1 << (self.note_strings[i] - 1))
                     and _can_assign(remaining, used | (1 << (self.note_strings[i] - 1))))


@dataclass
class GenerationResult:
    tokens: list[str]
    stopped_eos: bool
    metadata: dict


def prepare_encoder(tokenizer, tokens, max_input_tokens=512):
    """Match training's explicit EOS; never silently truncate."""
    tokens = list(tokens)
    if tokenizer.eos_token is None:
        raise UnsupportedInputError("Tokenizer must define EOS")
    if tokens and tokens[-1] == tokenizer.eos_token:
        tokens.pop()
    if tokenizer.eos_token in tokens:
        raise UnsupportedInputError("EOS is only legal at the end of input")
    vocab = tokenizer.get_vocab()
    unknown = [t for t in tokens if t not in vocab or t in tokenizer.all_special_tokens]
    if unknown:
        raise UnsupportedInputError(f"Unknown or special input tokens: {unknown[:3]}")
    encoded = tokenizer(" ".join(tokens + [tokenizer.eos_token]),
                        add_special_tokens=False, return_tensors="pt")
    if encoded["input_ids"][0].tolist() != [vocab[t] for t in tokens + [tokenizer.eos_token]]:
        raise UnsupportedInputError("Tokenizer must encode each domain token as exactly one ID")
    if encoded["input_ids"].shape[1] > max_input_tokens:
        raise UnsupportedInputError("Input exceeds budget including EOS; use complete-onset chunking")
    return encoded


def _reorder(cache, parents):
    if hasattr(cache, "reorder_cache"):
        cache.reorder_cache(parents)
        return cache
    return tuple(tuple(value.index_select(0, parents) for value in layer) for layer in cache)


def generate_tab(model, tokenizer, tokens, *, mode="anchored", beams=4,
                 max_input_tokens=512, max_new_tokens=512, no_repeat_ngram=3,
                 repetition_penalty=1.0, stochastic=False, max_fret=24):
    import torch
    if mode not in ("anchored", "legacy") or beams < 1:
        raise ValueError("Expected anchored/legacy mode and positive beam count")
    if mode == "anchored" and stochastic:
        raise ValueError("Anchored decoding is deterministic; stochastic mode is legacy-only")
    template = AnchoredTemplate(tokenizer, tokens, max_fret) if mode == "anchored" else None
    if template and len(template.slots) > max_new_tokens:
        raise UnsupportedInputError("Output budget cannot cover every input slot and EOS")
    encoded = {k: v.to(model.device) for k, v in prepare_encoder(tokenizer, tokens, max_input_tokens).items()}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            if template is None:
                kwargs = dict(max_new_tokens=max_new_tokens, num_beams=beams, do_sample=stochastic,
                              no_repeat_ngram_size=no_repeat_ngram or 0,
                              repetition_penalty=repetition_penalty)
                if beams > 1:
                    kwargs["early_stopping"] = True
                if stochastic:
                    kwargs.update(top_p=0.9, temperature=0.7)
                output = model.generate(**encoded, **kwargs)[0].tolist()
            else:
                # Own finite-beam expansion avoids infeasible/-inf dummy beams
                # when a fixed time slot or chord has fewer choices than width.
                encoder = model.get_encoder()(**encoded).last_hidden_state
                start = model.generation_config.decoder_start_token_id
                if start is None:
                    start = model.config.decoder_start_token_id
                if start is None:
                    raise UnsupportedInputError("Model must define decoder_start_token_id")
                prefixes, scores, cache = [[start]], [0.0], None
                for position, slot in enumerate(template.slots):
                    count = len(prefixes)
                    decoder_ids = torch.tensor([[p[-1]] for p in prefixes], device=model.device)
                    result = model(encoder_outputs=(encoder.expand(count, -1, -1),),
                                   attention_mask=encoded["attention_mask"].expand(count, -1),
                                   decoder_input_ids=decoder_ids, past_key_values=cache, use_cache=True)
                    probabilities = result.logits[:, -1].float().log_softmax(-1)
                    choices = []
                    for parent, prefix in enumerate(prefixes):
                        allowed = template.allowed(position, prefix)
                        values = probabilities[parent, list(allowed)].tolist()
                        choices.extend((scores[parent] + value, parent, token)
                                       for value, token in zip(values, allowed) if math.isfinite(value))
                    if not choices:
                        raise UnplayableInputError(f"No finite legal continuation at tick {slot.tick}")
                    choices.sort(key=lambda x: (-x[0], tuple(prefixes[x[1]]), x[2]))
                    selected = choices[:beams]
                    prefixes = [prefixes[parent] + [token] for _, parent, token in selected]
                    scores = [score for score, _, _ in selected]
                    if position != len(template.slots) - 1:
                        cache = _reorder(result.past_key_values,
                                         torch.tensor([parent for _, parent, _ in selected], device=model.device))
                output = prefixes[0]
    finally:
        model.train(was_training)
    generated = output[1:]
    stopped = tokenizer.eos_token_id in generated
    return GenerationResult(tokenizer.decode(output, skip_special_tokens=True).split(), stopped,
                            {"mode": mode, "beams": beams, "input_eos": True,
                             "no_repeat_ngram": 0 if template else no_repeat_ngram,
                             "preserves_input_attacks_and_time": template is not None,
                             "same_onset_unique_strings": template is not None,
                             "sustained_occupancy_enforced": False,
                             "generated_tokens": len(generated), "stopped_eos": stopped})


def chunk_complete_onsets(tokens, budget):
    """Raw MIDI chunks; TS stays with its following tick group for exact stitch."""
    if budget < 1:
        raise ValueError("Positive chunk budget required")
    groups, current = [], []
    for token in tokens:
        if token.startswith("TS_") and int(token[3:]) > 0 and current:
            groups.append(current)
            current = []
        current.append(token)
    if current:
        groups.append(current)
    chunks, current = [], []
    for group in groups:
        if len(group) > budget:
            raise UnsupportedInputError("A single simultaneous event group exceeds the input budget")
        if current and len(current) + len(group) > budget:
            chunks.append(current)
            current = []
        current.extend(group)
    if current:
        chunks.append(current)
    return chunks
