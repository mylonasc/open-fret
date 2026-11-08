"""Shared MIDI event helpers for training and inference."""

from __future__ import annotations


def midi_file_to_tokens(midi_path: str, target_resolution: int = 480):
    """Convert a MIDI file to NO/NF/TS tokens (no CAPO/TUNING prefix).

    Imported lazily so preprocessing tests don't require pretty_midi.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    res_scale = target_resolution / pm.resolution
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    if not notes:
        return []
    notes.sort(key=lambda n: n.start)
    events = []
    for note in notes:
        start_tick = int(round(pm.time_to_tick(note.start) * res_scale))
        end_tick = int(round(pm.time_to_tick(note.end) * res_scale))
        events.append((start_tick, "ON", note.pitch))
        events.append((end_tick, "OFF", note.pitch))
    events.sort(key=lambda x: (x[0], 0 if x[1] == "OFF" else 1))
    tokens: list[str] = []
    last_time: int | None = None
    for time, typ, pitch in events:
        if last_time is None:
            last_time = time
        else:
            delta = time - last_time
            if delta > 0:
                tokens.append(f"TS_{delta}")
                last_time = time
        tokens.append(f"NO_{pitch}" if typ == "ON" else f"NF_{pitch}")
    return tokens


def add_conditioning(tokens: list[str], capo: int = 0,
                     tuning: list[int] | None = None) -> list[str]:
    if tuning is None:
        tuning = [64, 59, 55, 50, 45, 40]
    tuning_str = "_".join(map(str, tuning))
    return [f"CAPO_{capo}", f"TUNING_{tuning_str}"] + list(tokens)
