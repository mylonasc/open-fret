"""Read-only event diagnostics: attacks are not the same as sustained polyphony."""

from collections import Counter, defaultdict
import re


def event_stream(tokens, kind, tuning=None, capo=0):
    """Cumulative TS ticks, including OFF-only steps; preserve duplicate notes."""
    if isinstance(tokens, str):
        tokens = tokens.split()
    events, tick = [], 0
    for token in tokens:
        if token.startswith("TS_"):
            delta = int(token[3:])
            if delta < 0:
                raise ValueError("Negative time shift")
            tick += delta
        elif kind == "midi" and token.startswith("NO_"):
            events.append({"tick": tick, "pitch": int(token[3:]), "string": None})
        elif kind == "tab" and token.startswith("TAB_"):
            parts = token.split("_")
            try:
                string, fret = int(parts[1]), int(parts[2])
            except (ValueError, IndexError):
                events.append({'tick': tick, 'pitch': None, 'string': 0})
                continue
            valid = 1 <= string <= len(tuning) and 0 <= fret <= 24 and tuning[string - 1] >= 0
            events.append({"tick": tick, "pitch": tuning[string - 1] + capo + fret if valid else None,
                           "string": string})
    return events, tick


def conditioning(tokens):
    tuning = next((list(map(int, t[7:].split("_"))) for t in tokens if t.startswith("TUNING_")), None)
    capo = next((int(t[5:]) for t in tokens if t.startswith("CAPO_")), None)
    if tuning is None or capo is None:
        raise ValueError("Explicit source tuning/capo required")
    return tuning, capo


def group_onsets(events):
    groups = defaultdict(list)
    for event in events:
        groups[event["tick"]].append(event)
    return dict(groups)


def structure(events):
    groups = group_onsets(events)
    sizes = [len(g) for g in groups.values()]
    return {"notes": len(events), "onsets": len(groups),
            "chord_onsets": sum(n > 1 for n in sizes),
            "chord_rate": sum(n > 1 for n in sizes) / max(len(groups), 1),
            "max_simultaneous_attacks": max(sizes, default=0),
            "same_string_conflict_onsets": sum(
                len(g) > len({e["string"] for e in g}) for g in groups.values()
                if g[0]["string"] is not None),
            "invalid_notes": sum(e["pitch"] is None for e in events),
            "valid_string_fret_rate": sum(e["pitch"] is not None for e in events) / max(len(events), 1),
            "duplicate_string_violations": sum(len(g) - len({e['string'] for e in g})
                                               for g in groups.values() if g[0]['string'] is not None)}


def timed_match(reference, predicted, tolerance=30):
    """Maximum-cardinality sorted matching per pitch, in ticks (480 PPQ)."""
    if tolerance < 0:
        raise ValueError('Onset tolerance must be nonnegative')
    ref_by_pitch, pred_by_pitch = defaultdict(list), defaultdict(list)
    for e in reference:
        ref_by_pitch[e["pitch"]].append(e["tick"])
    for e in predicted:
        if e["pitch"] is not None:
            pred_by_pitch[e["pitch"]].append(e["tick"])
    matched, errors = 0, []
    for pitch, ref in ref_by_pitch.items():
        ref = sorted(ref)
        pred = sorted(pred_by_pitch[pitch])
        i = j = 0
        while i < len(ref) and j < len(pred):
            if abs(ref[i] - pred[j]) <= tolerance:
                matched += 1
                errors.append(abs(ref[i] - pred[j]))
                i += 1
                j += 1
            elif pred[j] < ref[i]:
                j += 1
            else:
                i += 1
    precision = matched / len(predicted) if predicted else float(not reference)
    recall = matched / len(reference) if reference else float(not predicted)
    return {"tolerance_ticks": tolerance, "matched_notes": matched,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "matched_onset_mae_ticks": sum(errors) / len(errors) if errors else None}


def repetition_conflicts(tokens, n=3):
    seen, repeated = set(), []
    for end in range(n, len(tokens) + 1):
        gram = tuple(tokens[end - n:end])
        if gram in seen:
            repeated.append(end)
        seen.add(gram)
    return {"ngram_size": n, "forbidden_reference_positions": len(repeated),
            "reference_ngram_positions": max(0, len(tokens) - n + 1),
            "first_forbidden_token_1based": repeated[0] if repeated else None}


def compare_streams(input_tokens, ref_tokens, pred_tokens, tolerance=30):
    tuning, capo = conditioning(input_tokens)
    source, end = event_stream(input_tokens, "midi")
    reference, _ = event_stream(ref_tokens, "tab", tuning, capo)
    predicted, pred_end = event_stream(pred_tokens, "tab", tuning, capo)
    sg, pg = group_onsets(source), group_onsets(predicted)
    mono = [t for t, group in sg.items() if len(group) == 1]
    counts = Counter(e["pitch"] for e in source)
    pitch_overlap = sum((counts & Counter(e["pitch"] for e in predicted if e["pitch"] is not None)).values())
    same = Counter((e["tick"], e["pitch"]) for e in source) == Counter(
        (e["tick"], e["pitch"]) for e in reference)
    return {"input": structure(source), "reference": structure(reference), "prediction": structure(predicted),
            "input_reference_onsets_identical": same,
            "input_end_tick": end, "prediction_end_tick": pred_end,
            "note_count_ratio": len(predicted) / max(len(source), 1),
            "timed_pitch": timed_match(source, predicted, tolerance),
            "time_coverage_ratio": min(pred_end, end) / end if end else float(not predicted),
            "missing_attacks": max(0, len(source) - timed_match(source, predicted, tolerance)['matched_notes']),
            "extra_attacks": len(predicted) - timed_match(source, predicted, tolerance)['matched_notes'],
            "untimed_pitch_matched": pitch_overlap,
            "untimed_pitch_precision": pitch_overlap / max(len(predicted), 1),
            "untimed_pitch_recall": pitch_overlap / max(len(source), 1),
            "mono_reference_onsets": len(mono),
            "mono_onsets_with_any_prediction_exact_tick": sum(t in pg for t in mono),
            "false_chord_at_mono_onset_exact_tick": sum(len(pg.get(t, [])) > 1 for t in mono),
            "predicted_notes_after_input_end": sum(e["tick"] >= end for e in predicted),
            "prefix": pred_tokens[:20], "reference_repetition": repetition_conflicts(ref_tokens),
            "prediction_tokens": len(pred_tokens), "reference_tokens": len(ref_tokens),
            "at_512_token_budget": len(pred_tokens) == 512,
            "effective_note_vs_structural_coefficient_ratio":
                1 + (len(ref_tokens) + 1) / (0.3 * max(len(reference), 1))}


def family_key(group):
    """Heuristic version-family candidates, NOT proof of musical duplication."""
    artist, _, title = group.partition("__")
    title = re.sub(r"\.tex$", "", title, flags=re.I).replace("_", " ")
    title = re.sub(r"\s*\(\d+\)\s*$", "", title)
    clean = lambda x: re.sub(r"[^\w]+", "", x.casefold())
    return clean(artist), clean(title)
