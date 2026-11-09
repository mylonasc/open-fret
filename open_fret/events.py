"""Identity-preserving note sidecars; model tokens continue to encode onsets.

All ticks are absolute, integer 480 PPQ. A half-open interval [start, end)
occupies one string; a release at a tick precedes any attack at that tick.
"""
from collections import defaultdict, deque


def midi_token_notes(tokens, *, allow_open=False):
    """Pair NO/NF by FIFO for overlapping equal pitches in token-only inputs.

    MIDI files use their original note identities instead. An unterminated
    token-only attack is supported only by an explicit caller policy.
    """
    tick, notes, active = 0, [], defaultdict(deque)
    for token in tokens:
        if token.startswith("TS_"):
            delta = int(token[3:])
            if delta < 0:
                raise ValueError("Negative time shift")
            tick += delta
        elif token.startswith("NO_"):
            pitch = int(token[3:])
            if not 0 <= pitch <= 127:
                raise ValueError("Pitch outside MIDI range")
            note = {"id": str(len(notes)), "start": tick, "end": None, "pitch": pitch}
            notes.append(note)
            active[pitch].append(note)
        elif token.startswith("NF_"):
            pitch = int(token[3:])
            if not active[pitch]:
                raise ValueError(f"Unmatched release for pitch {pitch} at {tick}")
            active[pitch].popleft()["end"] = tick
        elif not token.startswith(("CAPO_", "TUNING_")) and token != "[EOS]":
            raise ValueError(f"Unsupported MIDI token: {token}")
    for note in notes:
        if note["end"] is None:
            if not allow_open:
                raise ValueError(f"Missing release for note {note['id']}")
            note["end"] = max(tick, note["start"] + 480)
            note["duration_policy"] = "open-note: end-of-stream or 480 ticks"
        if note["end"] <= note["start"]:
            raise ValueError("Zero/negative note duration")
    return notes


def notes_to_tokens(notes):
    """Stable pitch ordering and OFF-before-ON, preserving initial silence."""
    events = []
    for index, note in enumerate(notes):
        if note["start"] < 0 or note["end"] <= note["start"]:
            raise ValueError("Invalid note interval")
        events.extend([(note["start"], 1, note["pitch"], index),
                       (note["end"], 0, note["pitch"], index)])
    tick, result = 0, []
    for time, on, pitch, _ in sorted(events):
        if time > tick:
            result.append(f"TS_{time - tick}")
            tick = time
        result.append(f"{'NO' if on else 'NF'}_{pitch}")
    return result


def tab_attacks(tokens):
    tick, notes = 0, []
    for token in tokens:
        if token.startswith("TS_"):
            delta = int(token[3:])
            if delta < 0:
                raise ValueError("Negative time shift")
            tick += delta
        elif token.startswith("TAB_"):
            parts = token.split("_")
            if len(parts) not in (3, 4):
                raise ValueError(f"Malformed TAB token: {token}")
            notes.append({"start": tick, "string": int(parts[1]),
                          "fret": int(parts[2]), "tech": parts[3] if len(parts) == 4 else None})
        elif token != "[EOS]":
            raise ValueError(f"Unsupported TAB token: {token}")
    return notes


def validate_fingerings(notes, tuning, capo=0, max_fret=24):
    seen, active = set(), {}
    for note in sorted(notes, key=lambda n: (n["start"], str(n["id"]))):
        if note["id"] in seen:
            raise ValueError("Duplicate note identity")
        seen.add(note["id"])
        s, fret = note["string"], note["fret"]
        if not (1 <= s <= len(tuning) and 0 <= fret <= max_fret):
            raise ValueError(f"Invalid string/fret for note {note['id']}")
        if note["end"] <= note["start"] or note["start"] < 0:
            raise ValueError("Invalid note interval")
        if note["pitch"] != tuning[s - 1] + capo + fret:
            raise ValueError(f"Pitch mismatch for note {note['id']}")
        if active.get(s, 0) > note["start"]:
            raise ValueError(f"Occupied string {s} at tick {note['start']}")
        active[s] = note["end"]
