"""Token-level target curation shared by caching and inference diagnostics.

The corpus contains onset groups whose reference tab assigns two different
frets to the same string (overlapping voices or malformed source bars). Such a
group cannot be played and, more importantly, teaches a contradiction. Training
should ignore those groups rather than exclude an entire window: the affected
onsets are a small minority even though many windows contain one.

This module only inspects tokens; it never mutates source files.
"""

from __future__ import annotations


def _onset_groups(tab_tokens):
    """Yield ``[(index, string), ...]`` for each attack group."""
    group = []
    for index, token in enumerate(tab_tokens):
        if token.startswith("TS_"):
            if group:
                yield group
            group = []
        elif token.startswith("TAB_"):
            parts = token.split("_")
            if len(parts) < 3:
                raise ValueError(f"Malformed TAB token: {token}")
            group.append((index, int(parts[1])))
    if group:
        yield group


def conflict_positions(tab_tokens):
    """Indices of TAB tokens whose attack group repeats a string.

    The whole group is returned because keeping a subset would silently change
    the chord's pitch content. Order is ascending and duplicate-free.
    """
    result = []
    for group in _onset_groups(tab_tokens):
        strings = [string for _, string in group]
        if len(strings) > len(set(strings)):
            result.extend(index for index, _ in group)
    return result


def conflicts_for_window(midi_tokens, tuning, capo=0, max_fret=24):
    """Return MIDI onset pitches that cannot all fit on distinct strings.

    Uses the same candidate-domain logic as anchored decoding, so training
    excludes exactly the groups that constrained decoding must report as
    unsupported. Pitches outside tuning/capo/fret range are infeasible too.
    """
    from .generation import _can_assign
    if not tuning or any(not -1 <= p <= 127 for p in tuning):
        raise ValueError("Unsupported tuning")
    reachable = [p for p in tuning if p >= 0]
    allowed = [[p + capo + fret for fret in range(max_fret + 1)] for p in reachable]
    group = []
    bad = []

    def flush():
        if len(group) > 1:
            masks = []
            for pitch in group:
                mask = 0
                for string, pitches in enumerate(allowed):
                    if pitch in pitches:
                        mask |= 1 << string
                masks.append(mask)
            if not _can_assign(tuple(masks)):
                bad.extend(group)

    for token in midi_tokens:
        if token.startswith("TS_"):
            flush()
            group = []
        elif token.startswith("NO_"):
            group.append(int(token[3:]))
    flush()
    return bad
