"""AlphaTex parsing with explicit duration handling.

Fixes Issue #2: previously the string number in ``fret.string`` was
misread as a duration denominator (``0.1`` -> duration 1).

Supported note syntax (per note)::

    F.S[.D][...][tech]

where ``F`` is fret, ``S`` is string (1-based, 1 = high e),
``D`` is an optional explicit duration denominator (1,2,4,8,16,32,64),
``...`` are optional dots for dotted durations, and ``tech`` is an
optional technique suffix such as ``b4``, ``s`` or ``v``.

Examples: ``0.1``, ``0.1.8``, ``0.1.8.``, ``2.3b4``, ``2.3.8b4``.
Chords use ``(note note ...)`` with an optional outer ``.D`` duration.
Rests use ``r`` with optional ``.D``. Duration directives use ``:D``.
Inline ``.D`` persists, as in AlphaTab. ``{d}``/``{dd}`` dot just that beat.
Trailing dots and b/s/v suffixes are also supported as legacy dataset extensions.
Unsupported syntax raises rather than being scanned for plausible sub-tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TICKS_PER_QUARTER = 480
PARSER_VERSION = "alphatex-v2-strict-duration-inheritance"
ALPHATAB_PARSER_VERSION = "alphatab-1.3.0-midi-duration-v1"
SUPPORTED_PARSER_VERSIONS = (PARSER_VERSION, ALPHATAB_PARSER_VERSION)
DEFAULT_TUNING_MIDI = [64, 59, 55, 50, 45, 40]  # E5 B4 G4 D4 A3 E3 (dataset convention) = concert MIDI
DEFAULT_TEMPO = 120

_NOTE_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}

_TUNING_RE = re.compile(r"([A-G][#b]?)(-?\d+)")
_DURATION_DIRECTIVE_RE = re.compile(r":([0-9]+)(\.*)")
_REST_RE = re.compile(r"^r(?:\.([0-9]+))?(\.*)$")
# F.S + optional .D + dots + optional tech. Trailing dots after tech allowed.
_SINGLE_NOTE_RE = re.compile(
    r"^(\d+)\.(\d+)(?:\.(\d+))?(\.*)([bsv][0-9]*)?(\.*)$"
)
_OUTER_DURATION_RE = re.compile(r"\.([0-9]+)(\.*)$")
_MASTER_RE = re.compile(
    r"(?P<dur>:[0-9]+\.*)"
    r"|(?P<chord>\([^\)]+\)(?:\.[0-9]+\.*)?)"
    r"|(?P<note>[0-9]+\.[0-9]+(?:\.[0-9]+)?\.*(?:[bsv][0-9]*)?\.*)"
    r"|(?P<rest>r(?:\.[0-9]+\.*)?)"
    r"|(?P<bar>\|)"
)


class AlphaTexParseError(ValueError):
    """Unsupported or malformed input; never return a partially parsed track."""


class StaleParserDataError(ValueError):
    """Cached/processed supervision predates the current parser contract."""


@dataclass
class ParsedNote:
    fret: int
    string: int
    tech: str | None = None


@dataclass
class TimedEvent:
    start: int
    duration: int
    notes: list[ParsedNote] = field(default_factory=list)


@dataclass
class ParsedTrack:
    tuning: list[int]
    tempo: int
    events: list[TimedEvent] = field(default_factory=list)


def parse_tuning(tuning_str: str) -> list[int]:
    """Convert e.g. ``E5 B4 G4 D4 A3 E3`` to MIDI numbers (high-e first).

    Uses the dataset/upstream convention (see ``_parse_tuning`` in
    vladsavelyev/guitartab): ``value = octave * 12 + note`` where C=0..B=11,
    so standard tuning ``E5 B4 G4 D4 A3 E3`` maps to ``[64, 59, 55, 50, 45, 40]``
    (concert MIDI). NOTE: this is one octave above scientific pitch notation
    names (middle C = C5 here); do NOT add 1 to the octave.
    """
    midi_numbers: list[int] = []
    for token in tuning_str.split():
        m = _TUNING_RE.fullmatch(token)
        if not m:
            raise AlphaTexParseError(f"Invalid tuning note: {token!r}")
        name, octave = m.groups()
        midi_numbers.append(int(octave) * 12 + _NOTE_MAP[name])
    return midi_numbers


def ticks_for_duration(duration: int, dots: int = 0,
                       ticks_per_quarter: int = TICKS_PER_QUARTER) -> int:
    if duration not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError(f"Invalid duration denominator: {duration}")
    base = int(ticks_per_quarter * (4 / duration))
    total = base
    multiplier = 0.5
    for _ in range(dots):
        total += int(base * multiplier)
        multiplier *= 0.5
    return total


def parse_single_note(token: str) -> tuple[ParsedNote, int | None, int]:
    """Parse ``F.S[.D][...][tech]``.

    Returns (note, explicit_duration_or_None, dots).
    """
    m = _SINGLE_NOTE_RE.match(token)
    if not m:
        raise ValueError(f"Unsupported note token: {token!r}")
    fret_s, string_s, dur_s, dots1, tech, dots2 = m.groups()
    explicit: int | None = int(dur_s) if dur_s else None
    dots = len(dots1 or "") + len(dots2 or "")
    return ParsedNote(fret=int(fret_s), string=int(string_s), tech=tech or None), explicit, dots


def _split_chord_inner(inner: str) -> list[str]:
    return [p for p in inner.strip().split() if p]


class AlphaTexParser:
    def __init__(self, ticks_per_quarter: int = TICKS_PER_QUARTER):
        self.ticks_per_quarter = ticks_per_quarter

    def parse_text(self, content: str) -> list[ParsedTrack]:
        tempo_match = re.search(r"\\tempo (\d+)", content)
        tempo = int(tempo_match.group(1)) if tempo_match else DEFAULT_TEMPO
        tracks_raw = content.split("\\track")
        parsed: list[ParsedTrack] = []
        for t_raw in tracks_raw[1:]:
            # The remainder of the \track line is a name, not beat syntax.
            # In particular, digits/dots in a title must never become notes.
            t_raw = t_raw.partition("\n")[2]
            tuning = list(DEFAULT_TUNING_MIDI)
            tuning_match = re.search(r"\\tuning ([^\n\\]+)", t_raw)
            if tuning_match:
                parsed_tuning = parse_tuning(tuning_match.group(1).strip())
                if parsed_tuning:
                    tuning = parsed_tuning
            events = self._parse_track_body(t_raw)
            for event in events:
                if any(not 1 <= n.string <= len(tuning) for n in event.notes):
                    raise AlphaTexParseError("Note string outside the declared tuning")
            parsed.append(ParsedTrack(tuning=tuning, tempo=tempo, events=events))
        return parsed

    def parse_file(self, path: str) -> list[ParsedTrack]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse_text(content)

    def _parse_track_body(self, t_raw: str) -> list[TimedEvent]:
        current_duration = 4
        current_dots = 0
        lines: list[str] = []
        metadata_seen: set[str] = set()
        for line in t_raw.split("\n"):
            stripped = line.strip()
            if not stripped or stripped == ".":
                continue
            if stripped.startswith("\\"):
                # Only non-timing track metadata can be ignored safely.
                name = stripped.split()[0]
                if name not in ("\\tuning", "\\instrument", "\\frets", "\\clef"):
                    raise AlphaTexParseError(f"Unsupported track directive: {stripped!r}")
                if lines or name in metadata_seen:
                    raise AlphaTexParseError(f"Repeated or mid-track metadata: {stripped!r}")
                if name in ("\\instrument", "\\frets") and not re.fullmatch(r"\\\w+\s+\d+", stripped):
                    raise AlphaTexParseError(f"Unsupported metadata value: {stripped!r}")
                metadata_seen.add(name)
                continue
            lines.append(line)
        clean_content = " ".join(lines)

        time_tokens: list[TimedEvent] = []
        current_ticks = 0
        cursor = 0
        while cursor < len(clean_content):
            if clean_content[cursor].isspace():
                cursor += 1
                continue
            m = _MASTER_RE.match(clean_content, cursor)
            if m is None:
                raise AlphaTexParseError(
                    f"Unsupported syntax at offset {cursor}: {clean_content[cursor:cursor+60]!r}")
            cursor = m.end()
            if cursor < len(clean_content) and not (
                    clean_content[cursor].isspace() or clean_content[cursor] in "|{"):
                raise AlphaTexParseError(f"Malformed token near {clean_content[m.start():cursor+30]!r}")
            kind = m.lastgroup
            token = m.group(0).strip()
            if not token:
                continue
            if kind == "dur":
                dm = _DURATION_DIRECTIVE_RE.match(token)
                if dm:
                    current_duration = int(dm.group(1))
                    current_dots = len(dm.group(2) or "")
                continue
            if kind == "bar":
                continue
            # AlphaTex dotted beat effects; all other effects (including tuplets,
            # ties, repeats and tempo changes) must be reported, not stripped.
            effect_dots = None
            effect = re.match(r"\s*\{\s*(d|dd)\s*\}", clean_content[cursor:])
            if effect:
                effect_dots = len(effect.group(1))
                cursor += effect.end()
            if kind == "rest":
                rm = _REST_RE.match(token)
                dur = int(rm.group(1)) if rm and rm.group(1) else current_duration
                dots = len(rm.group(2) or "") if rm else 0
                if rm and rm.group(1):
                    dots = len(rm.group(2) or "")
                else:
                    dots = current_dots if not (rm and rm.group(2)) else len(rm.group(2))
                    if rm and rm.group(2):
                        dots = len(rm.group(2))
                # If rest has no explicit duration/dots, inherit directive.
                if not (rm and (rm.group(1) or rm.group(2))):
                    dur, dots = current_duration, current_dots
                if rm and rm.group(1):
                    current_duration, current_dots = dur, 0
                if effect_dots is not None:
                    dots = effect_dots
                current_ticks += ticks_for_duration(dur, dots, self.ticks_per_quarter)
                continue
            # chord or single note
            notes: list[ParsedNote] = []
            explicit_dur: int | None = None
            dots = 0
            if kind == "chord":
                # Separate outer duration: "(...).D..."
                inner = token
                outer_dur: int | None = None
                outer_dots = 0
                if token.endswith(")") is False:
                    om = _OUTER_DURATION_RE.search(token)
                    if om:
                        outer_dur = int(om.group(1))
                        outer_dots = len(om.group(2) or "")
                        inner = token[: om.start()]
                # inner should be "( ... )"
                inner_content = inner[inner.find("(") + 1: inner.rfind(")")]
                for part in _split_chord_inner(inner_content):
                    note, note_dur, note_dots = parse_single_note(part)
                    if note_dur is not None or note_dots:
                        raise AlphaTexParseError("Chord duration/dots must follow the closing parenthesis")
                    notes.append(note)
                    # Keep first explicit duration if no outer duration.
                    if explicit_dur is None and note_dur is not None:
                        explicit_dur = note_dur
                        dots = note_dots
                if outer_dur is not None:
                    explicit_dur = outer_dur
                    dots = outer_dots
            else:
                note, note_dur, note_dots = parse_single_note(token)
                notes = [note]
                explicit_dur = note_dur
                dots = note_dots

            if not notes:
                continue
            dur = explicit_dur if explicit_dur is not None else current_duration
            use_dots = dots if (explicit_dur is not None or dots) else current_dots
            # If token had no explicit duration and no trailing dots, inherit directive dots.
            if explicit_dur is None and dots == 0:
                use_dots = current_dots
            if explicit_dur is not None:
                # Like AlphaTab's _currentDuration, inline .D persists until
                # another inline duration or :D directive changes it.
                current_duration, current_dots = explicit_dur, 0
            if effect_dots is not None:
                use_dots = effect_dots
            actual_ticks = ticks_for_duration(dur, use_dots, self.ticks_per_quarter)
            # Filter invalid strings/frets but keep event if at least one valid note.
            # Validation against tuning length happens in sequence builder.
            time_tokens.append(TimedEvent(start=current_ticks, duration=actual_ticks, notes=notes))
            current_ticks += actual_ticks
        return time_tokens
