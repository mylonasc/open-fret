"""Tablature rendering (Issue #7)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re

STANDARD_TUNING_STR = "E5 B4 G4 D4 A3 E3"

# General MIDI program number (0-indexed) used for the tab track.
# 24 = Acoustic Guitar (nylon) — the classical sound. (Was 29/Overdrive.)
NYLON_GUITAR_PROGRAM = 24

# Sounding-pitch equivalent of STANDARD_TUNING_STR (see
# TabRenderer.sounding_tuning): what the AlphaTex header must carry so tab
# playback lands on concert pitch in AlphaTab.
SOUNDING_TUNING_STR = "E4 B3 G3 D3 A2 E2"

@dataclass(frozen=True)
class DurationPiece:
    ticks: int
    denominator: int
    dots: int = 0
    tuplet: int = 1

    def effects(self):
        values = (["d" * self.dots] if self.dots else [])
        if self.tuplet != 1:
            values.append(f"tu {self.tuplet}")
        return "{" + " ".join(values) + "}" if values else ""


# Scientific-pitch names indexed by MIDI pitch class; AlphaTab maps written
# note names literally (E5 -> MIDI 76), so MIDI m spells PITCH_NAMES[m % 12]
# in octave m // 12 - 1 (60 -> C4).
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_note_name(midi: int) -> str:
    """Spell a MIDI pitch as AlphaTex pitched-track note text (``C#4``)."""
    if not isinstance(midi, int) or isinstance(midi, bool) or not 0 <= midi <= 127:
        raise ValueError(f"Unsupported MIDI pitch: {midi!r}")
    octave = midi // 12 - 1
    if not 0 <= octave <= 9:
        raise ValueError(f"MIDI pitch out of notatable octave range: {midi}")
    return f"{PITCH_NAMES[midi % 12]}{octave}"


def _duration_specs():
    result = {}
    ratios = {1: 1, 3: 2, 5: 4, 6: 4, 7: 4, 9: 8, 10: 8, 11: 8, 12: 8}
    for denominator in (-4, -2, 1, 2, 4, 8, 16, 32, 64, 128, 256):
        # Match AlphaTab 1.3.0's integer arithmetic at its native 960 PPQ.
        base = int(3840 / denominator) if denominator > 0 else 3840 * -denominator
        for dots in range(3):
            dotted = base if dots == 0 else base + (base // 2 if dots == 1 else (base // 4) * 3)
            for tuplet, ratio in ratios.items():
                native = dotted * ratio // tuplet
                if native % 2 or native <= 0:
                    continue  # Never introduce fractional 480-PPQ ticks.
                piece = DurationPiece(native // 2, denominator, dots, tuplet)
                old = result.get(piece.ticks)
                rank = lambda p: (p.tuplet != 1, p.dots, abs(p.denominator), p.tuplet)
                if old is None or rank(piece) < rank(old):
                    result[piece.ticks] = piece
    return result


_DURATION_SPECS = _duration_specs()


def _duration_token(delta: int) -> str | None:
    """Exact recipe only. Effects belong AFTER the beat, never in ':8.'."""
    if delta in _DURATION_SPECS:
        piece = _DURATION_SPECS[delta]
        return f":{piece.denominator}" + (" " + piece.effects() if piece.effects() else "")
    return None


@lru_cache(maxsize=4096)
def duration_pieces(ticks: int) -> tuple[DurationPiece, ...]:
    """Exact additive notation; unrepresentable ticks fail, never approximate."""
    if ticks <= 0 or ticks > 1_000_000:
        raise ValueError(f"Unsupported duration: {ticks} ticks")
    if ticks in _DURATION_SPECS:
        return (_DURATION_SPECS[ticks],)
    coins = sorted(_DURATION_SPECS, reverse=True)
    # Emission is bar-bounded (<=1920); bound this public helper too.
    if ticks > 3840:
        whole = (ticks - 1920) // 1920
        remainder = ticks - whole * 1920
        return (_DURATION_SPECS[1920],) * whole + (duration_pieces(remainder) if remainder else ())
    counts = [0] + [ticks + 1] * ticks
    previous = [None] * (ticks + 1)
    for value in range(1, ticks + 1):
        for coin in coins:
            if coin <= value and counts[value - coin] + 1 < counts[value]:
                counts[value] = counts[value - coin] + 1
                previous[value] = coin
    if previous[ticks] is None:
        raise ValueError(f"Cannot represent exactly {ticks} ticks in AlphaTex at 480 PPQ")
    pieces = []
    while ticks:
        coin = previous[ticks]
        pieces.append(_DURATION_SPECS[coin])
        ticks -= coin
    return tuple(pieces)


class TabRenderer:
    def __init__(self, tempo: int = 120, tuning_str: str = STANDARD_TUNING_STR):
        self.tempo = tempo
        self.tuning_str = tuning_str

    def to_alphatex_notes(self, notes, title="open-fret"):
        """Render identity-bearing intervals, including independent sustains.

        Partition at every attack/release and tie continuing notes individually
        inside chords. A release never reattacks a surviving note. This accepts
        source-aligned sidecars, not invented note ends from TAB/TS tokens.
        """
        from .alphatex import parse_tuning
        from .events import validate_fingerings
        notes = list(notes)
        validate_fingerings(notes, parse_tuning(self.tuning_str))
        points = sorted({0} | {int(n[k]) for n in notes for k in ('start', 'end')})
        body, tick = [], 0
        for start, end in zip(points, points[1:]):
            active = sorted((n for n in notes if n['start'] <= start < n['end']),
                            key=lambda n: n['string'])
            remaining = end - start
            while remaining:
                if tick and tick % 1920 == 0:
                    body.append('|')
                segment = min(remaining, 1920 - tick % 1920)
                for piece in duration_pieces(segment):
                    chord = []
                    for n in active:
                        if n['start'] < tick:
                            chord.append(f"-.{n['string']}")
                        else:
                            tech = f"_{n['tech']}" if n.get('tech') else ''
                            formatted, _ = self._collect_chords([f"TAB_{n['string']}_{n['fret']}{tech}"])
                            chord.append(formatted[0][0])
                    beat = self.format_chord(chord) if chord else 'r'
                    body.append(f":{piece.denominator} {beat}{piece.effects()}")
                    tick += piece.ticks
                    remaining -= piece.ticks
        return '\n'.join([f'\\title {json.dumps(str(title), ensure_ascii=False)}',
                          f'\\tempo {self.tempo}', '.', '\\track "Tab"',
                          f'\\instrument {NYLON_GUITAR_PROGRAM}',
                          f'\\tuning {self.sounding_tuning(self.tuning_str)}',
                          ' '.join(body)]) + '\n'

    @staticmethod
    def sounding_tuning(tuning_str: str) -> str:
        """Written pitch -> sounding pitch (one octave down).

        Guitar music is written an octave above concert pitch; AlphaTab maps
        note names literally (E5 -> MIDI 76), so the header must carry the
        sounding octave (E4 -> MIDI 64) for tab audio to match the MIDI.
        """
        import re

        def down_octave(tok: str) -> str:
            return re.sub(r"(\d+)$",
                          lambda m: str(int(m.group(1)) - 1), tok)
        return " ".join(down_octave(t) for t in tuning_str.split())

    def to_alphatex_midi(self, notes, title="open-fret") -> str:
        """Render exact input notes as a pitched track for A/B listening.

        Same partition/timing as :meth:`to_alphatex_notes` (attacks preserved,
        sustains tied per voice), but spelled as concert pitches on an
        untuned staff instead of frets. The viewer plays this through the
        same AlphaTab player as the tab so the two can be compared without
        a second clock. Tied continuations use the ``{t}`` effect, which
        AlphaTab resolves by sounding pitch (verified against 1.3.0).
        """
        notes = list(notes)
        for n in notes:
            if (not isinstance(n.get("start"), int) or isinstance(n.get("start"), bool)
                    or not isinstance(n.get("end"), int) or isinstance(n.get("end"), bool)
                    or not n["start"] < n["end"]):
                raise ValueError(f"Input notes need integer start < end: {n!r}")
            midi_note_name(n["pitch"])  # fail loudly on unnotatable pitches
        points = sorted({0} | {int(n[k]) for n in notes for k in ('start', 'end')})
        body, tick = [], 0
        for start, end in zip(points, points[1:]):
            active = sorted((n for n in notes if n['start'] <= start < n['end']),
                            key=lambda n: n['pitch'])
            remaining = end - start
            while remaining:
                if tick and tick % 1920 == 0:
                    body.append('|')
                segment = min(remaining, 1920 - tick % 1920)
                for piece in duration_pieces(segment):
                    chord = []
                    for n in active:
                        name = midi_note_name(n['pitch'])
                        chord.append(f"{name}{{t}}" if n['start'] < tick else name)
                    beat = self.format_chord(chord) if chord else 'r'
                    body.append(f":{piece.denominator} {beat}{piece.effects()}")
                    tick += piece.ticks
                    remaining -= piece.ticks
        return '\n'.join([f'\\title {json.dumps(str(title), ensure_ascii=False)}',
                          f"\\tempo {self.tempo}", ".",
                          '\\track "Input MIDI"',
                          # Notation-only staff: without this AlphaTab lays out a
                          # (stringless) TAB staff and crashes painting it.
                          "\\staff {score}",
                          f"\\instrument {NYLON_GUITAR_PROGRAM}",
                          ' '.join(body)]) + '\n'

    def to_alphatex(self, tokens, title="open-fret") -> str:
        """Render attacks exactly; absent final duration defaults to 480 ticks.

        TAB/TS encodes onsets, not independent note ends. Here each onset group
        sustains until the next group/end tick. Ties preserve attacks when a
        duration crosses bars or needs multiple notation fragments. Recovering
        actual independent sustains requires source note identities/durations.
        """
        if isinstance(tokens, str):
            tokens = tokens.split()
        self._validate_tokens(tokens)
        chords, gaps = self._collect_chords(tokens)
        # Header format mirrors upstream guitartab (vladsavelyev) output:
        # quoted title/track names are required — bare \track is rejected
        # by AlphaTab's importer ("invalid value: 'null'").
        lines = [f'\\title {json.dumps(str(title), ensure_ascii=False)}', f"\\tempo {self.tempo}", ".",
                 '\\track "Tab"', f"\\instrument {NYLON_GUITAR_PROGRAM}",
                 f"\\tuning {self.sounding_tuning(self.tuning_str)}"]
        body: list[str] = []
        tick = 0
        for chord_notes, gap in zip(chords, gaps):
            if not 0 < gap <= 1_000_000:
                raise ValueError(f"Unsupported gap: {gap}")
            remaining, continuation = gap, False
            while remaining:
                if tick and tick % 1920 == 0:
                    body.append("|")
                segment = min(remaining, 1920 - tick % 1920)
                for piece in duration_pieces(segment):
                    notes = chord_notes
                    if continuation and notes:
                        notes = [f"-.{note.split('.')[1].split('{')[0]}" for note in notes]
                    beat = self.format_chord(notes) if notes else "r"
                    body.append(f":{piece.denominator} {beat}{piece.effects()}")
                    tick += piece.ticks
                    remaining -= piece.ticks
                    continuation = bool(chord_notes)
        lines.append(" ".join(body))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _collect_chords(tokens: list[str]) -> tuple[list[list[str]], list[int]]:
        """Group TABs into chords; gap = TS sum until next chord (or default)."""
        chords: list[list[str]] = []
        gaps: list[int] = []
        cur: list[str] = []
        pending = 0  # TS accumulated since last TAB (gap after previous chord)
        for t in tokens:
            if t.startswith("TS_"):
                try:
                    pending += int(t.split("_")[1])
                except ValueError as e:
                    raise ValueError(f"Invalid time shift: {t}") from e
                if int(t.split("_")[1]) < 0:
                    raise ValueError(f"Negative time shift: {t}")
            elif t.startswith("TAB_"):
                parts = t.split("_")
                if len(parts) not in (3, 4):
                    raise ValueError(f"Invalid TAB token: {t}")
                s, f = parts[1], parts[2]
                if int(s) < 1 or int(f) < 0:
                    raise ValueError(f"Invalid string/fret: {t}")
                tech = parts[3] if len(parts) > 3 else ""
                effect = ""
                if tech:
                    if re.fullmatch(r"b\d+", tech):
                        effect = "{b (0 " + tech[1:] + ")}"
                    elif tech in ("s", "v"):
                        effect = "{" + tech + "}"
                    else:
                        raise ValueError(f"Unsupported technique: {tech}")
                if cur and pending > 0:
                    # Gap since previous TAB -> previous chord complete.
                    chords.append(cur)
                    gaps.append(pending)
                    cur = []
                    pending = 0
                elif not cur and pending > 0:
                    chords.append([])
                    gaps.append(pending)
                    pending = 0
                cur.append(f"{f}.{s}{effect}")
            elif t != '[EOS]':
                raise ValueError(f"Unsupported tab token: {t}")
        if cur:
            gaps.append(pending if pending > 0 else 480)
            chords.append(cur)
        elif pending > 0:
            chords.append([])
            gaps.append(pending)
        return chords, gaps

    @staticmethod
    def format_chord(note_strs: list[str]) -> str:
        if len(note_strs) == 1:
            return note_strs[0]
        return "(" + " ".join(note_strs) + ")"

    def _validate_tokens(self, tokens):
        from .alphatex import parse_tuning
        from .events import tab_attacks
        count = len(parse_tuning(self.tuning_str))
        used = set()
        for note in tab_attacks(tokens):
            if not 1 <= note['string'] <= count or not 0 <= note['fret'] <= 24:
                raise ValueError("Tab string/fret is outside supported tuning/range")
            key = (note['start'], note['string'])
            if key in used:
                raise ValueError("Same-onset string conflict cannot be rendered faithfully")
            used.add(key)

    def to_ascii(self, tokens) -> str:
        if isinstance(tokens, str):
            tokens = tokens.split()
        self._validate_tokens(tokens)
        self._collect_chords(tokens)  # validate rather than silently dropping syntax
        from .alphatex import parse_tuning
        tuning = parse_tuning(self.tuning_str)
        strings: dict[int, list[str]] = {i: [] for i in range(1, len(tuning) + 1)}
        # Build columns: each onset (chord) is one column across all strings.
        cur_chord: dict[int, str] = {}
        pending_dashes = 0

        def flush_chord():
            nonlocal pending_dashes
            if not cur_chord:
                # Pure gap: extend all strings with dashes.
                if pending_dashes > 0:
                    for s in strings:
                        strings[s].append("-" * pending_dashes)
                    pending_dashes = 0
                return
            width = max([len(v) for v in cur_chord.values()] + [1])
            width = max(width, 1)
            for s in strings:
                cell = cur_chord.get(s, "-")
                strings[s].append(cell.ljust(width, "-"))
            if pending_dashes > 0:
                for s in strings:
                    strings[s].append("-" * pending_dashes)
                pending_dashes = 0
            cur_chord.clear()

        for t in tokens:
            if t.startswith("TS_"):
                try:
                    delta = int(t.split("_")[1])
                except ValueError:
                    continue
                if delta == 0:
                    continue  # same onset, not an ASCII time advance
                # Flush previous chord before counting gap after it.
                if cur_chord:
                    flush_chord()
                pending_dashes += max(1, delta // 60)
            elif t.startswith("TAB_"):
                if not cur_chord and pending_dashes:
                    flush_chord()  # leading/inter-onset silence occupies shared columns
                parts = t.split("_")
                if len(parts) < 3:
                    continue
                try:
                    s_idx = int(parts[1])
                except ValueError:
                    continue
                fret = parts[2]
                tech = parts[3] if len(parts) > 3 else ""
                if s_idx in cur_chord:
                    # Same string twice without TS: keep first, append second
                    # as extra char (conflict should be fixed upstream).
                    cur_chord[s_idx] += f"{fret}{tech}"
                elif s_idx in strings:
                    cur_chord[s_idx] = f"{fret}{tech}"
        flush_chord()
        names = ['e', 'B', 'G', 'D', 'A', 'E'] if tuning == [64, 59, 55, 50, 45, 40] else [str(i) for i in strings]
        width = max(map(len, names))
        return "\n".join(f"{names[i-1].rjust(width)}|{''.join(strings[i])}" for i in strings)
