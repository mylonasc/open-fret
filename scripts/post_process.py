"""Backwards-compatible wrapper around open_fret.postprocess (Issue #8)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.postprocess import (  # noqa: F401
    NeighborSearchCorrector,
    CorrectionReport,
    neighbor_search,
    get_pitch_from_tab as _unused,  # placeholder for API compat
)


def get_pitch_from_tab(s, f, tuning):
    from open_fret.postprocess import pitch_from_tab
    result = pitch_from_tab(s, f, list(tuning))
    return result if result is not None else -1


def post_process_alphatex(tokens_str, input_midi_tokens):
    tokens = tokens_str.split() if isinstance(tokens_str, str) else tokens_str
    corrected = neighbor_search(tokens, input_midi_tokens)
    return " ".join(corrected)
