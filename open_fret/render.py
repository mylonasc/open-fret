"""Tablature rendering (Issue #7)."""

from __future__ import annotations

STANDARD_TUNING_STR = "E5 B4 G4 D4 A3 E3"

_DUR_MAP = {
    1920: ":1", 1440: ":1.", 960: ":2", 720: ":2.",
    480: ":4", 360: ":4.", 240: ":8", 180: ":8.",
    120: ":16", 90: ":16.", 60: ":32", 30: ":64",
}


def _duration_token(delta: int) -> str | None:
    if delta in _DUR_MAP:
        return _DUR_MAP[delta]
    best = min(_DUR_MAP, key=lambda k: abs(k - delta))
    if abs(best - delta) / max(delta, 1) <= 0.25:
        return _DUR_MAP[best]
    return None


class TabRenderer:
    def __init__(self, tempo: int = 120, tuning_str: str = STANDARD_TUNING_STR):
        self.tempo = tempo
        self.tuning_str = tuning_str

    def to_alphatex(self, tokens) -> str:
        if isinstance(tokens, str):
            tokens = tokens.split()
        chords, gaps = self._collect_chords(tokens)
        parts_out = [f"\\tempo {self.tempo}", "\\track", f"\\tuning {self.tuning_str}", "."]
        for chord_notes, gap in zip(chords, gaps):
            dur_tok = _duration_token(gap) if gap > 0 else ":4"
            if dur_tok is None:
                dur_tok = ":4"
            parts_out.append(dur_tok)
            parts_out.append(self.format_chord(chord_notes))
        return " ".join(parts_out)

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
                except ValueError:
                    continue
            elif t.startswith("TAB_"):
                parts = t.split("_")
                if len(parts) < 3:
                    continue
                s, f = parts[1], parts[2]
                tech = parts[3] if len(parts) > 3 else ""
                if cur and pending > 0:
                    # Gap since previous TAB -> previous chord complete.
                    chords.append(cur)
                    gaps.append(pending)
                    cur = []
                    pending = 0
                elif not cur and pending > 0:
                    # Leading silence before first chord: ignore for duration.
                    pending = 0
                cur.append(f"{f}.{s}{tech}")
        if cur:
            gaps.append(pending if pending > 0 else 480)
            chords.append(cur)
        return chords, gaps

    @staticmethod
    def format_chord(note_strs: list[str]) -> str:
        if len(note_strs) == 1:
            return note_strs[0]
        return "(" + " ".join(note_strs) + ")"

    def to_ascii(self, tokens) -> str:
        if isinstance(tokens, str):
            tokens = tokens.split()
        strings: dict[int, list[str]] = {i: [] for i in range(1, 7)}
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
            for s in range(1, 7):
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
                # Flush previous chord before counting gap after it.
                if cur_chord:
                    flush_chord()
                pending_dashes += max(1, delta // 60)
            elif t.startswith("TAB_"):
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
        names = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}
        return "\n".join(f"{names[i]}|{''.join(strings[i])}" for i in range(1, 7))
