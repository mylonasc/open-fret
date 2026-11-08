"""Versioned vocabulary semantics for pitch-aware supervision (epic #14)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


METADATA_VERSION = "tab-pitch-v2-source-conditioning"
MAX_STRINGS = 12
_TAB = re.compile(r"TAB_(\d+)_(\d+)(?:_(.+))?")


@dataclass(frozen=True)
class TokenMetadata:
    tokens: tuple[str, ...]
    max_fret: int = 24

    @classmethod
    def from_tokenizer(cls, tokenizer, max_fret=24):
        vocab = tokenizer.get_vocab()
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError("Loss metadata requires a contiguous tokenizer vocabulary")
        tokens = [""] * len(vocab)
        for token, index in vocab.items():
            tokens[index] = token
        return cls(tuple(tokens), max_fret)

    def __post_init__(self):
        if not 0 <= self.max_fret <= 127:
            raise ValueError("max_fret must be between 0 and 127")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary tokens must be unique")

    def note(self, token_id):
        """Return written string (high string = 1), fret and technique, or None."""
        match = _TAB.fullmatch(self.tokens[token_id])
        if match is None:
            return None
        string, fret, technique = match.groups()
        return int(string), int(fret), technique

    @staticmethod
    def tuning(token):
        if not token.startswith("TUNING_"):
            return None
        try:
            values = tuple(int(x) for x in token[len("TUNING_"):].split("_"))
        except ValueError:
            raise ValueError(f"Malformed tuning token: {token!r}") from None
        # -1 marks unavailable strings in some corpus tunings. It is never
        # a sounding pitch and must not produce an equivalent candidate.
        if not values or all(x == -1 for x in values) or any(not -1 <= x <= 127 for x in values):
            raise ValueError(f"Invalid concert-MIDI tuning: {token!r}")
        return values

    def pitch(self, token_id, tuning, capo=0):
        """Concert pitch; tuning is uncapoed and string 1 is the high string."""
        note = self.note(token_id)
        if note is None:
            return None
        string, fret, _ = note
        if not 1 <= string <= len(tuning) or not 0 <= fret <= self.max_fret:
            return None
        if tuning[string - 1] < 0:
            return None
        pitch = tuning[string - 1] + capo + fret
        return pitch if 0 <= pitch <= 127 else None

    @staticmethod
    def capo(token):
        if not token.startswith("CAPO_"):
            return None
        try:
            value = int(token[len("CAPO_"):])
        except ValueError:
            raise ValueError(f"Malformed capo token: {token!r}") from None
        if not 0 <= value <= 24:
            raise ValueError(f"Invalid capo: {token!r}")
        return value

    def manifest(self):
        spec = {"version": METADATA_VERSION, "tokens": self.tokens,
                "max_fret": self.max_fret,
                "pitch_convention": "concert-MIDI; tuning excludes capo"}
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        return {"version": METADATA_VERSION, "sha256": digest,
                "vocab_size": len(self.tokens), "max_fret": self.max_fret,
                "pitch_convention": spec["pitch_convention"]}
