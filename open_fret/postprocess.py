"""Timed, identity-preserving correction with sustained string occupancy.

Correction is explicit and separately reported: it is not a raw model metric.
The interval assignment is bounded search, not a claim of general biomechanics.
"""
from dataclasses import dataclass, field
from collections import Counter
from .events import midi_token_notes, tab_attacks, validate_fingerings

DEFAULT_TUNING = [64, 59, 55, 50, 45, 40]
MAX_FRET = 24


def pitch_from_tab(string, fret, tuning=DEFAULT_TUNING):
    return tuning[string - 1] + fret if 1 <= string <= len(tuning) and fret >= 0 else None


def viable_string_fret(target_pitch, prefer_string=None, tuning=DEFAULT_TUNING, max_fret=MAX_FRET):
    candidates = [(s, target_pitch - pitch) for s, pitch in enumerate(tuning, 1)
                  if 0 <= target_pitch - pitch <= max_fret]
    return min(candidates, key=lambda sf: (sf[0] != prefer_string, sf[1], sf[0])) if candidates else None


@dataclass
class CorrectionReport:
    inserted: int = 0
    pitch_fixed: int = 0
    refingered: int = 0
    dropped_extra: int = 0
    same_string_conflicts_fixed: int = 0
    unplayable: int = 0
    raw_notes: int = 0
    corrected_notes: list[dict] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sustained_occupancy_enforced: bool = True


class CorrectionError(ValueError):
    def __init__(self, message, report):
        report.unresolved.append(message)
        self.report = report
        super().__init__(message)


class NeighborSearchCorrector:
    def __init__(self, tuning=None, max_fret=MAX_FRET, capo=0, max_states=100000):
        self.tuning = list(tuning) if tuning is not None else list(DEFAULT_TUNING)
        self.max_fret, self.capo, self.max_states = max_fret, capo, max_states

    def correct(self, pred_tokens, input_midi_tokens, *, source_notes=None):
        from .token_metadata import TokenMetadata
        pred_tokens = pred_tokens.split() if isinstance(pred_tokens, str) else list(pred_tokens)
        midi = input_midi_tokens.split() if isinstance(input_midi_tokens, str) else list(input_midi_tokens)
        tuning, capo = self.tuning, self.capo
        for token in midi:
            if token.startswith("TUNING_"):
                tuning = TokenMetadata.tuning(token)
            elif token.startswith("CAPO_"):
                capo = TokenMetadata.capo(token)
        if not tuning or capo is None or not 0 <= capo <= 24:
            raise ValueError("Invalid tuning/capo")
        source = list(source_notes) if source_notes is not None else midi_token_notes(midi, allow_open=True)
        source = sorted(source, key=lambda n: (n["start"], n["pitch"], str(n["id"])))
        if source_notes is not None:
            tick, attacks = 0, []
            for token in midi:
                if token.startswith('TS_'):
                    delta = int(token[3:])
                    if delta < 0:
                        raise ValueError('Negative source time shift')
                    tick += delta
                elif token.startswith('NO_'):
                    attacks.append((tick, int(token[3:])))
            if Counter(attacks) != Counter((n['start'], n['pitch']) for n in source):
                raise ValueError('Source note sidecar attacks differ from MIDI input')
        if len({n['id'] for n in source}) != len(source):
            raise ValueError("Duplicate source note identity")
        if any(n['start'] < 0 or n['end'] <= n['start'] for n in source):
            raise ValueError("Invalid source note interval")
        predictions = tab_attacks(pred_tokens)
        report = CorrectionReport(raw_notes=len(predictions))
        if any(n.get("duration_policy") for n in source):
            report.notes.append("Token-only missing releases: explicit end-of-stream/480-tick policy; not original durations")
        # Exact tick matching, never pitch-order matching across attacks.
        used, preferred, domains = set(), [], []
        for note in source:
            candidates = [i for i, p in enumerate(predictions)
                          if i not in used and p['start'] == note['start']]
            def distance(i):
                p = predictions[i]
                pitch = pitch_from_tab(p['string'], p['fret'], tuning)
                return abs((pitch + capo) - note['pitch']) if pitch is not None else 10000
            match = min(candidates, key=lambda i: (distance(i), i)) if candidates else None
            preferred.append(predictions[match] if match is not None else None)
            if match is not None:
                used.add(match)
            else:
                report.inserted += 1
            domain = [(s, note['pitch'] - base - capo) for s, base in enumerate(tuning, 1)
                      if 0 <= note['pitch'] - base - capo <= self.max_fret]
            p = preferred[-1]
            domain.sort(key=lambda sf: (not p or sf != (p['string'], p['fret']), sf[1], sf[0]))
            if not domain:
                report.unplayable += 1
                raise CorrectionError(f"No legal fingering for note {note['id']} pitch {note['pitch']}", report)
            domains.append(domain)
        report.dropped_extra = len(predictions) - len(used)
        # Iterative DFS across the complete song, not independent per-chord
        # greedy repair. Backtracking can refinger a preceding sustained note.
        assigned, choices, index, states = [], [0] * len(source), 0, 0
        while index < len(source):
            if states >= self.max_states:
                raise CorrectionError("Fingering search budget exceeded; no partial output emitted", report)
            states += 1
            selected = None
            while choices[index] < len(domains[index]):
                s, f = domains[index][choices[index]]
                choices[index] += 1
                if not any(old['string'] == s and old['end'] > source[index]['start'] for old in assigned):
                    selected = {**source[index], 'string': s, 'fret': f, 'tech': None}
                    break
            if selected is None:
                choices[index] = 0
                if not assigned:
                    report.unplayable += 1
                    raise CorrectionError("No sustained-string assignment for supplied note intervals", report)
                assigned.pop()
                index -= 1
            else:
                assigned.append(selected)
                index += 1
        for note, pred in zip(assigned, preferred):
            if pred is not None:
                raw_pitch = pitch_from_tab(pred['string'], pred['fret'], tuning)
                same_pitch = raw_pitch is not None and raw_pitch + capo == note['pitch']
                report.pitch_fixed += not same_pitch
                changed = (note['string'], note['fret']) != (pred['string'], pred['fret'])
                report.refingered += changed
                if same_pitch:
                    note['tech'] = pred['tech']
                elif pred['tech']:
                    report.notes.append(f"Dropped pitch-dependent technique on corrected note {note['id']}")
        for i, p in enumerate(predictions):
            report.same_string_conflicts_fixed += any(q['start'] == p['start'] and q['string'] == p['string']
                                                       for q in predictions[:i])
        validate_fingerings(assigned, tuning, capo, self.max_fret)
        report.corrected_notes = assigned
        # Preserve the full source time axis, including note-off-only ticks.
        by_tick = {}
        for note in assigned:
            by_tick.setdefault(note['start'], []).append(note)
        tick, output = 0, []
        for time in sorted(set(by_tick) | {n['end'] for n in assigned}):
            if time > tick:
                output.append(f"TS_{time - tick}")
            tick = time
            for note in by_tick.get(time, []):
                effect = f"_{note['tech']}" if note.get('tech') else ''
                output.append(f"TAB_{note['string']}_{note['fret']}{effect}")
        return output, report


def neighbor_search(pred_tokens, input_midi_tokens, tuning=DEFAULT_TUNING):
    corrected, _ = NeighborSearchCorrector(tuning=tuning).correct(pred_tokens, input_midi_tokens)
    return corrected
