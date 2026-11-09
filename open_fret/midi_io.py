"""Shared MIDI event helpers for training and inference."""

from __future__ import annotations


def read_midi_notes(midi_path: str, target_resolution: int = 480, instrument=None):
    """Convert a MIDI file to NO/NF/TS tokens (no CAPO/TUNING prefix).

    Imported lazily so preprocessing tests don't require pretty_midi.
    """
    import mido
    from collections import defaultdict, deque
    midi = mido.MidiFile(midi_path)
    if midi.type == 2:
        raise ValueError("Asynchronous type-2 MIDI is unsupported")
    scale = target_resolution / midi.ticks_per_beat
    groups = {}
    tempo_events = []
    for track_index, track in enumerate(midi.tracks):
        tick, ordinal, programs = 0, 0, defaultdict(int)
        active = defaultdict(deque)
        for msg in track:
            tick += msg.time
            if msg.type == 'set_tempo':
                tempo_events.append((tick, msg.tempo))
            elif msg.type == 'program_change':
                programs[msg.channel] = msg.program
            elif msg.type == 'control_change' and msg.control == 64 and msg.value >= 64:
                if msg.channel != 9:
                    raise ValueError("Sustain-pedal MIDI requires pedal-aware note ends; unsupported")
            elif msg.type == 'pitchwheel' and msg.pitch and msg.channel != 9:
                raise ValueError("Pitch-bend MIDI is unsupported by fixed-pitch note alignment")
            elif msg.type in ('note_on', 'note_off'):
                if msg.channel == 9:
                    continue
                if msg.type == 'note_on' and msg.velocity:
                    key = (track_index, msg.channel, programs[msg.channel])
                    note = {'id': f'{track_index}:{msg.channel}:{ordinal}',
                            'start': int(round(tick * scale)), 'end': None, 'pitch': msg.note}
                    ordinal += 1
                    groups.setdefault(key, []).append(note)
                    active[msg.channel, msg.note].append(note)
                else:
                    queue = active[msg.channel, msg.note]
                    if not queue:
                        raise ValueError(f"Unmatched MIDI note-off in track {track_index}")
                    queue.popleft()['end'] = int(round(tick * scale))
        if any(active.values()):
            raise ValueError(f"Unterminated MIDI notes in track {track_index}")
    observed_tempi = {tempo for _, tempo in tempo_events}
    if tempo_events and min(tick for tick, _ in tempo_events) > 0:
        observed_tempi.add(500000)  # implicit tempo before the first event
    if len(observed_tempi) > 1:
        raise ValueError("Variable-tempo MIDI is unsupported by single-tempo tab exports")
    tempo = next(iter(observed_tempi), 500000)
    keys = sorted(groups)
    eligible = list(range(len(keys)))
    if instrument is None:
        if len(eligible) > 1:
            raise ValueError(f"Multiple pitched instruments {eligible}; select --instrument INDEX")
        selected = eligible
    else:
        if instrument not in eligible:
            raise ValueError(f"Instrument {instrument} has no supported pitched notes")
        selected = [instrument]
    notes = []
    for i in selected:
        for note in groups[keys[i]]:
            if note['end'] <= note['start']:
                raise ValueError("MIDI note rounds to zero/negative duration at 480 PPQ")
            notes.append(note)
    notes.sort(key=lambda n: (n["start"], n["pitch"], n["id"]))
    return notes, {"tempo": float(mido.tempo2bpm(tempo)), "ticks_per_beat": target_resolution,
                   "instrument": selected[0] if selected else None,
                   "instruments": [{'index': i, 'track': key[0], 'channel': key[1], 'program': key[2]}
                                   for i, key in enumerate(keys)],
                   "equal_pitch_release_policy": "FIFO within track/channel/pitch",
                   "drum_policy": "channel 10 ignored",
                   "ignored_instruments": [i for i in eligible if i not in selected]}


def midi_file_to_tokens(midi_path: str, target_resolution: int = 480, instrument=None):
    from .events import notes_to_tokens
    notes, _ = read_midi_notes(midi_path, target_resolution, instrument)
    return notes_to_tokens(notes)


def add_conditioning(tokens: list[str], capo: int = 0,
                     tuning: list[int] | None = None) -> list[str]:
    if tuning is None:
        tuning = [64, 59, 55, 50, 45, 40]
    tuning_str = "_".join(map(str, tuning))
    return [f"CAPO_{capo}", f"TUNING_{tuning_str}"] + list(tokens)
