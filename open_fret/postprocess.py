"""Pitch/playability post-processing (Issue #8).

Replaces the order-only matcher with a timing-aware corrector that:
- aligns predictions to target MIDI onsets by musical order,
- fixes playable fret/string while preserving pitch when possible,
- inserts missing notes and drops unmatchable extras explicitly,
- detects simultaneous same-string conflicts within a chord.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_TUNING = [64, 59, 55, 50, 45, 40]
MAX_FRET = 24


def pitch_from_tab(string: int, fret: int, tuning: list[int] = DEFAULT_TUNING) -> int | None:
    if 1 <= string <= len(tuning) and fret >= 0:
        return tuning[string - 1] + fret
    return None


def viable_string_fret(target_pitch: int, prefer_string: int | None = None,
                       tuning: list[int] = DEFAULT_TUNING,
                       max_fret: int = MAX_FRET) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for s in range(1, len(tuning) + 1):
        f = target_pitch - tuning[s - 1]
        if 0 <= f <= max_fret:
            candidates.append((s, f))
    if not candidates:
        return None
    if prefer_string is not None:
        for s, f in candidates:
            if s == prefer_string:
                return (s, f)
    # Prefer lowest fret (most playable) then lowest string movement.
    candidates.sort(key=lambda sf: (sf[1], abs(sf[0] - (prefer_string or sf[0]))))
    return candidates[0]


@dataclass
class ParsedPred:
    token_index: int
    string: int
    fret: int
    tech: str | None
    pitch: int | None
    chord_id: int


@dataclass
class CorrectionReport:
    inserted: int = 0
    pitch_fixed: int = 0
    dropped_extra: int = 0
    same_string_conflicts_fixed: int = 0
    unplayable: int = 0
    notes: list[str] = field(default_factory=list)


def _parse_tab_token(token: str) -> tuple[int, int, str | None] | None:
    if not token.startswith("TAB_"):
        return None
    parts = token.split("_")
    if len(parts) < 3:
        return None
    try:
        s = int(parts[1])
        f = int(parts[2])
    except ValueError:
        return None
    tech = parts[3] if len(parts) > 3 else None
    return s, f, tech


def _chord_ids(tokens: list[str]) -> list[int]:
    """Assign chord ids: TABs between TS boundaries share an id."""
    ids: list[int] = []
    chord = 0
    for t in tokens:
        ids.append(chord)
        if t.startswith("TS_"):
            chord += 1
    return ids


class NeighborSearchCorrector:
    def __init__(self, tuning: list[int] | None = None, max_fret: int = MAX_FRET):
        self.tuning = list(tuning) if tuning else list(DEFAULT_TUNING)
        self.max_fret = max_fret

    def correct(self, pred_tokens: list[str],
                input_midi_tokens: list[str]) -> tuple[list[str], CorrectionReport]:
        report = CorrectionReport()
        target_pitches: list[int] = []
        for t in input_midi_tokens:
            if t.startswith("NO_"):
                try:
                    target_pitches.append(int(t.split("_")[1]))
                except ValueError:
                    continue
        chord_ids = _chord_ids(pred_tokens)
        preds: list[ParsedPred] = []
        for idx, tok in enumerate(pred_tokens):
            parsed = _parse_tab_token(tok)
            if parsed is None:
                continue
            s, f, tech = parsed
            pitch = pitch_from_tab(s, f, self.tuning)
            preds.append(ParsedPred(token_index=idx, string=s, fret=f,
                                    tech=tech, pitch=pitch, chord_id=chord_ids[idx]))

        output = list(pred_tokens)
        used: set[int] = set()

        # Align each target onset in order to the nearest unused prediction
        # within a small window (paper's +/-5), falling back to insertion.
        for target_idx, target_pitch in enumerate(target_pitches):
            best = None
            best_diff = None
            start = max(0, target_idx - 5)
            end = min(len(preds), target_idx + 6)
            # If lengths differ greatly, search all unused.
            if len(preds) != len(target_pitches):
                start, end = 0, len(preds)
            for p_idx in range(start, end):
                if p_idx in used:
                    continue
                p = preds[p_idx]
                if p.pitch is None:
                    diff = 10 ** 9
                else:
                    diff = abs(p.pitch - target_pitch)
                if best is None or diff < best_diff:
                    best = p_idx
                    best_diff = diff
                    if diff == 0:
                        break
            if best is None:
                # No unused prediction left: insert.
                inserted = self._insert_token(target_pitch, None)
                if inserted is None:
                    report.unplayable += 1
                    report.notes.append(f"unplayable pitch {target_pitch}")
                else:
                    output.append(inserted)
                    report.inserted += 1
                continue
            used.add(best)
            p = preds[best]
            if p.pitch != target_pitch:
                fixed = viable_string_fret(target_pitch, prefer_string=p.string,
                                           tuning=self.tuning, max_fret=self.max_fret)
                if fixed is None:
                    report.unplayable += 1
                    continue
                ns, nf = fixed
                tech_sfx = f"_{p.tech}" if p.tech else ""
                output[p.token_index] = f"TAB_{ns}_{nf}{tech_sfx}"
                report.pitch_fixed += 1
                p.string, p.fret = ns, nf
                p.pitch = target_pitch
            # Validate range even when pitch already matched.
            if not (1 <= p.string <= len(self.tuning) and 0 <= p.fret <= self.max_fret):
                fixed = viable_string_fret(target_pitch, prefer_string=None,
                                           tuning=self.tuning, max_fret=self.max_fret)
                if fixed is None:
                    report.unplayable += 1
                else:
                    ns, nf = fixed
                    tech_sfx = f"_{p.tech}" if p.tech else ""
                    output[p.token_index] = f"TAB_{ns}_{nf}{tech_sfx}"
                    report.pitch_fixed += 1

        # Drop predictions that were never aligned to a target.
        extra = [i for i in range(len(preds)) if i not in used]
        # Rebuild output without extras: mark for removal by index.
        if extra and len(preds) > len(target_pitches):
            remove_token_idx = {preds[i].token_index for i in extra}
            output = [t for idx, t in enumerate(output) if idx not in remove_token_idx]
            report.dropped_extra += len(remove_token_idx)
            # Re-parse chord ids after removal for conflict detection.
            chord_ids = _chord_ids(output)

        # Fix simultaneous same-string conflicts within each chord.
        report.same_string_conflicts_fixed += self._fix_chord_conflicts(output, target_pitches)
        return output, report

    def _insert_token(self, pitch: int, prefer_string: int | None) -> str | None:
        fixed = viable_string_fret(pitch, prefer_string=prefer_string,
                                   tuning=self.tuning, max_fret=self.max_fret)
        if fixed is None:
            return None
        s, f = fixed
        return f"TAB_{s}_{f}"

    def _fix_chord_conflicts(self, tokens: list[str], target_pitches: list[int]) -> int:
        fixed_count = 0
        # Group TAB indices by chord.
        chord_ids = _chord_ids(tokens)
        by_chord: dict[int, list[int]] = {}
        for idx, tok in enumerate(tokens):
            if tok.startswith("TAB_"):
                by_chord.setdefault(chord_ids[idx], []).append(idx)
        for chord, indices in by_chord.items():
            seen: dict[int, int] = {}
            for tok_idx in indices:
                parsed = _parse_tab_token(tokens[tok_idx])
                if parsed is None:
                    continue
                s, f, tech = parsed
                if s in seen:
                    # Conflict: same string twice in one chord.
                    pitch = pitch_from_tab(s, f, self.tuning)
                    alt = None
                    if pitch is not None:
                        # Find alternative string producing same pitch.
                        for cand_s in range(1, len(self.tuning) + 1):
                            if cand_s == s:
                                continue
                            cand_f = pitch - self.tuning[cand_s - 1]
                            if 0 <= cand_f <= self.max_fret:
                                alt = (cand_s, cand_f)
                                break
                    if alt is not None:
                        ns, nf = alt
                        tech_sfx = f"_{tech}" if tech else ""
                        tokens[tok_idx] = f"TAB_{ns}_{nf}{tech_sfx}"
                        fixed_count += 1
                    # else: leave as-is; caller reports unplayable separately.
                else:
                    seen[s] = tok_idx
        return fixed_count


def neighbor_search(pred_tokens, input_midi_tokens, tuning=DEFAULT_TUNING):
    """Backwards-compatible wrapper returning only corrected tokens."""
    corrector = NeighborSearchCorrector(tuning=list(tuning))
    midi_list = input_midi_tokens if isinstance(input_midi_tokens, list) else str(input_midi_tokens).split()
    pred_list = pred_tokens if isinstance(pred_tokens, list) else str(pred_tokens).split()
    corrected, _ = corrector.correct(pred_list, midi_list)
    return corrected
