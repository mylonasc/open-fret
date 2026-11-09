"""Complete-song decoding with immutable source-note identity sidecars."""
from dataclasses import asdict

from .events import notes_to_tokens
from .generation import chunk_complete_onsets, generate_tab
from .midi_io import add_conditioning
from .postprocess import NeighborSearchCorrector


def transcribe_notes(model, tokenizer, notes, *, tuning=None, capo=0,
                     max_input_tokens=512, max_new_tokens=512, beams=4,
                     decoding_mode="anchored", stochastic=False):
    tuning = list(tuning or [64, 59, 55, 50, 45, 40])
    tokens = notes_to_tokens(notes)
    if not notes:
        raise ValueError("No pitched notes in selected MIDI instrument")
    budget = min(max_input_tokens - 3, max_new_tokens - 1)
    if budget < 1:
        raise ValueError("Budgets must accommodate conditioning, events and EOS")
    raw, windows, offset = [], [], 0
    for index, chunk in enumerate(chunk_complete_onsets(tokens, budget)):
        result = generate_tab(model, tokenizer, add_conditioning(chunk, capo, tuning),
                              mode=decoding_mode, beams=beams,
                              max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
                              stochastic=stochastic)
        raw.extend(result.tokens)
        end = offset + sum(int(t[3:]) for t in chunk if t.startswith('TS_'))
        windows.append({"index": index, "origin_tick": offset, "end_tick": end,
                        **result.metadata})
        offset = end
    corrected, report = NeighborSearchCorrector(tuning=tuning, capo=capo).correct(
        raw, add_conditioning(tokens, capo, tuning), source_notes=notes)
    from .transcription_diagnostics import compare_streams
    conditioned = add_conditioning(tokens, capo, tuning)
    return {"schema_version": "inference-v1", "source_notes": list(notes),
            "raw_tokens": raw, "corrected_tokens": corrected,
            "correction": asdict(report), "windows": windows,
            "raw_metrics": compare_streams(conditioned, raw, corrected),
            "corrected_metrics": compare_streams(conditioned, corrected, corrected),
            "tuning": tuning, "capo": capo,
            "limitations": ["Fingering feasibility is not a biomechanical guarantee",
                            "Note-end identity comes from MIDI, not model output",
                            "Pitch-dependent techniques on changed pitches are dropped"]}
