"""Generated musical evaluation, frozen provenance and family-cluster uncertainty.

Validation selects; held-out test is final reporting only. No dataset is silently
substituted and failed generations stay in denominators.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re

from .events import midi_token_notes, tab_attacks
from .generation import generate_tab
from .postprocess import NeighborSearchCorrector
from .splits import canonical_family, group_id_from_filename
from .transcription_diagnostics import conditioning, compare_streams, event_stream, structure


def sequence_metrics(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, pred in enumerate(prediction, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j-1] + (ref != pred)))
        previous = current
    return {'edit_distance': previous[-1], 'reference_tokens': len(reference),
            'prediction_tokens': len(prediction), 'exact_sequence': reference == prediction,
            'normalized_edit_error': previous[-1] / max(1, len(reference))}


def row_profile(data, name):
    tuning, capo = conditioning(data['midi'])
    events, end = event_stream(data['midi'], 'midi')
    stats = structure(events)
    pitches = [e['pitch'] for e in events]
    bigrams = list(zip(pitches, pitches[1:]))
    riff = any(n > 1 for n in Counter(bigrams).values())
    overlap, active = False, Counter()
    for t in data['midi']:
        if t.startswith('NF_'):
            active[int(t[3:])] -= 1
        elif t.startswith('NO_'):
            overlap |= sum(max(0, n) for n in active.values()) > 0
            active[int(t[3:])] += 1
    style = ('chord' if stats['chord_onsets'] else 'riff' if riff else
             'arpeggio_proxy' if overlap else 'mono')
    frets = [n['fret'] for n in tab_attacks(data['tab'])]
    source = data.get('source') or data.get('file') or group_id_from_filename(name)
    return {'source': source, 'family': canonical_family(source), 'input_notes': len(events),
            'strata': [style, f"domain:{data.get('instrument_domain', 'unknown')}",
                       'tuning:standard' if tuning == [64, 59, 55, 50, 45, 40] else 'tuning:alternate',
                       f'capo:{capo}', f'strings:{len(tuning)}', 'length:long' if end > 7680 else 'length:short',
                       'reference_fret:high' if any(f >= 15 for f in frets) else 'reference_fret:low'],
            'style_definition': 'arpeggio_proxy = overlapping releases with monophonic attacks; not a manual genre label'}


def freeze_validation_panel(root, manifest_path, sources=512, seed=42):
    root, path = Path(root), Path(manifest_path)
    candidates = [n.strip() for n in (root / 'val_files.txt').read_text().splitlines() if n.strip()]
    if not candidates or len(candidates) != len(set(candidates)) or sources < 1:
        raise ValueError('Validation list must be nonempty/unique and sources positive')
    split_hash = hashlib.sha256('\n'.join(candidates).encode()).hexdigest()
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get('schema_version') != 'validation-panel-v2':
            raise ValueError('Panel schema differs; preserve existing evidence and use a new panel path')
        if (manifest['split_sha256'] != split_hash or manifest['seed'] != seed
                or manifest['requested_sources'] != sources):
            raise ValueError('Frozen validation panel identity differs; use a new panel path')
        for row in manifest['rows']:
            content = (root / 'individual' / row['file']).read_bytes()
            if row['file'] not in candidates or hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError('Frozen panel source changed')
        return manifest
    groups = defaultdict(list)
    for name in candidates:
        group = group_id_from_filename(name)
        if group == Path(name).stem:
            row = json.loads((root / 'individual' / name).read_text())
            group = row.get('source') or row.get('file') or group
        groups[group].append(name)
    rng = random.Random(seed)
    pool = []
    for group in sorted(groups):
        name = rng.choice(sorted(groups[group]))
        content = (root / 'individual' / name).read_bytes()
        data = json.loads(content)
        profile = row_profile(data, name)
        pool.append({'file': name, 'sha256': hashlib.sha256(content).hexdigest(), **profile})
    by_tag = defaultdict(list)
    for i, row in enumerate(pool):
        for tag in row['strata']:
            by_tag[tag].append(i)
    selected = []
    for tag in sorted(by_tag, key=lambda t: (len(by_tag[t]), t)):
        if len(selected) >= sources:
            break
        if not any(tag in pool[i]['strata'] for i in selected):
            selected.append(rng.choice(by_tag[tag]))
    remaining = [i for i in range(len(pool)) if i not in selected]
    selected.extend(rng.sample(remaining, min(max(0, sources - len(selected)), len(remaining))))
    rng.shuffle(selected)
    rows = [pool[i] for i in selected]
    manifest = {'schema_version': 'validation-panel-v2', 'split': 'validation',
                'split_sha256': split_hash, 'seed': seed, 'requested_sources': sources,
                'source_groups_available': len(groups),
                'selection': 'one uniform window per source; observed-stratum coverage floors then uniform sources; no feasibility/capo/domain exclusions',
                'stratification': 'coverage-first over observed source representatives; not a population-unbiased corpus estimate',
                'available_strata': {tag: len(indices) for tag, indices in by_tag.items()},
                'family_policy': 'canonical named-version clusters; heuristic uncertainty units, not proof of deduplication',
                'test_music_read': False, 'rows': rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(manifest, indent=2))
    temp.replace(path)
    return manifest


def minimum_fret_baseline(midi):
    """No learned weights; onset-group duration policy explicitly not sustains."""
    tuning, capo = conditioning(midi)
    events, end = event_stream(midi, 'midi')
    ticks = sorted({e['tick'] for e in events} | {end})
    source = [{'id': str(i), 'start': e['tick'], 'pitch': e['pitch'],
               'end': next((tick for tick in ticks if tick > e['tick']), e['tick'] + 480)}
              for i, e in enumerate(events)]
    tokens, _ = NeighborSearchCorrector(tuning, capo=capo).correct([], midi, source_notes=source)
    return tokens


def evaluate_row(model, tokenizer, data, *, modes=('legacy', 'anchored'),
                 beams=4, max_input_tokens=512, max_new_tokens=512, tolerance=30):
    midi, reference = data['midi'], data['tab']
    output = {}
    def scored(tokens, metadata):
        metrics = compare_streams(midi, reference, tokens, tolerance=tolerance)
        tempo = data.get('tempo', 120)
        metrics['timed_pitch']['tolerance_seconds'] = tolerance / 480 * 60 / tempo
        mae = metrics['timed_pitch']['matched_onset_mae_ticks']
        metrics['timed_pitch']['matched_onset_mae_seconds'] = mae / 480 * 60 / tempo if mae is not None else None
        metrics['timed_pitch']['seconds_basis'] = 'source tempo' if 'tempo' in data else 'assumed 120 BPM (source tempo missing)'
        metrics.pop('effective_note_vs_structural_coefficient_ratio', None)
        metrics.pop('at_512_token_budget', None)
        metrics['sequence'] = sequence_metrics(reference, tokens)
        metrics['tokens'] = tokens
        metrics['malformed_tokens'] = sum(not re.fullmatch(r'TS_\d+|TAB_\d+_\d+(?:_(?:[sv]|b\d+))?|\[EOS\]', t)
                                           for t in tokens)
        metrics.update(metadata)
        return metrics
    try:
        output['baseline'] = scored(minimum_fret_baseline(midi),
                                    {'kind': 'minimum-fret no-learning, onset-duration policy'})
    except ValueError as e:
        output['baseline'] = {'error': str(e)}
    for mode in modes:
        try:
            generated = generate_tab(model, tokenizer, midi, mode=mode, beams=beams,
                                      max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
                                      no_repeat_ngram=0)
            output[mode] = scored(generated.tokens, generated.metadata)
            output[mode]['premature_eos'] = generated.stopped_eos and output[mode]['missing_attacks'] > 0
            output[mode]['cap_stop'] = not generated.stopped_eos
            try:
                # Prefer sidecars. A training window can contain a release of a
                # carried note; never infer its original identity/end from tokens.
                if data.get('carry_notes'):
                    raise ValueError('Carried-note occupancy requires complete-song context; no window-only correction claim')
                if data.get('notes'):
                    origin = data.get('start_tick', 0)
                    source = [{**n, 'start': n['start'] - origin, 'end': n['end'] - origin} for n in data['notes']]
                else:
                    source = midi_token_notes(midi)
                corrected, report = NeighborSearchCorrector().correct(generated.tokens, midi, source_notes=source)
                output[mode + '_corrected'] = scored(corrected, {'correction': asdict(report),
                    'duration_scope': 'attacks owned by this window; carry occupancy requires full-song context'})
            except ValueError as e:
                output[mode + '_corrected'] = {'error': str(e), 'scope': 'no duration/occupancy claim on incomplete window'}
        except (ValueError, RuntimeError) as e:
            output[mode] = {'error': str(e)}
            output[mode + '_corrected'] = {'error': 'generation unavailable'}
    return output


def summarize_rows(rows, seed=42, bootstrap_samples=400):
    modes = sorted({mode for row in rows for mode in row.get('generations', {})})
    def aggregate(selected, mode):
        attempts = [row for row in selected if mode in row.get('generations', {})]
        failed = sum('error' in row['generations'][mode] for row in attempts)
        values = [row['generations'][mode].get('timed_pitch', {}).get('f1', 0.) for row in attempts]
        return {'attempted_sources': len(attempts), 'failed_sources': failed,
                'macro_f1_failures_as_zero': sum(values) / len(values) if values else None,
                'macro_f1_success_only': sum(values) / (len(values) - failed) if len(values) > failed else None,
                'input_notes_including_failures': sum(row['input_notes'] for row in attempts)}
    summary = {mode: aggregate(rows, mode) for mode in modes}
    families = defaultdict(list)
    for row in rows:
        families[row['family']].append(row)
    rng = random.Random(seed)
    for mode in modes:
        estimates = []
        keys = sorted(key for key, members in families.items()
                      if any(mode in r.get('generations', {}) for r in members))
        summary[mode]['attempted_family_clusters'] = len(keys)
        if len(keys) >= 2:
            for _ in range(bootstrap_samples):
                sample = [r for key in rng.choices(keys, k=len(keys)) for r in families[key]]
                value = aggregate(sample, mode)['macro_f1_failures_as_zero']
                if value is not None:
                    estimates.append(value)
        estimates.sort()
        summary[mode]['family_cluster_bootstrap_95_percent'] = (
            [estimates[int(.025 * len(estimates))], estimates[min(len(estimates)-1, int(.975 * len(estimates)))]]
            if estimates else None)
    strata = sorted({tag for row in rows for tag in row['strata']})
    return {'audited_rows': len(rows), 'family_clusters': len(families), 'modes': summary,
            'strata': {tag: {mode: aggregate([r for r in rows if tag in r['strata']], mode) for mode in modes}
                       for tag in strata},
            'uncertainty': 'exploratory family-cluster bootstrap; small/heuristic families are not independent proof',
            'interpretation': 'anchored pitch F1 is constrained input fidelity, not learned fingering quality; five demos are qualitative'}
