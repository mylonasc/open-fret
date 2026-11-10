"""Read-only loss/data audit for a cached dataset.

Checks label validity and event balance over every row, then independently
recomputes the configured objective on real validation batches. Never writes
to the cache and never evaluates held-out test music.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.losses import LossConfig, TabLoss
from open_fret.token_metadata import TokenMetadata


def fast_label_stats(ds, metadata, eos_token_id, step=20000):
    """Bincount all labels without materializing the whole split."""
    import numpy as np
    vocab = len(metadata.tokens)
    counts = np.zeros(vocab, dtype=np.int64)
    invalid = rows = tokens = 0
    for start in range(0, len(ds), step):
        batch = ds[start:start + step]["labels"]
        arr = np.asarray(batch, dtype=np.int64).reshape(-1)
        valid = arr != -100
        values = arr[valid]
        invalid += int(((values < 0) | (values >= vocab)).sum())
        counts += np.bincount(values[(values >= 0) & (values < vocab)], minlength=vocab)
        tokens += int(valid.sum())
        rows += len(batch)
    classes = Counter()
    for token_id in np.nonzero(counts)[0]:
        token = metadata.tokens[int(token_id)]
        n = int(counts[token_id])
        if metadata.note(int(token_id)) is not None:
            classes['note'] += n
        elif token.startswith('TS_'):
            classes['time'] += n
        elif int(token_id) == eos_token_id:
            classes['eos'] += n
        else:
            classes['other'] += n
    classes['windows'] = rows
    return {'tokens': tokens, 'invalid_labels': invalid, 'classes': dict(classes)}


def conflict_stats(ds, metadata, limit):
    """Same-onset/same-string target conflicts; expensive, so bounded."""
    rows = conflict_windows = conflict_groups_total = duplicate_notes = 0
    for row in range(min(limit, len(ds))):
        labels = ds[row]["labels"]
        seq = [t for t in labels if t != -100]
        group = []
        groups = []
        for token_id in seq:
            token = metadata.tokens[token_id]
            if token.startswith('TS_'):
                if group:
                    groups.append(group)
                group = []
            elif metadata.note(token_id) is not None:
                group.append(metadata.note(token_id)[0])
        if group:
            groups.append(group)
        bad = [g for g in groups if len(g) > len(set(g))]
        rows += 1
        conflict_windows += bool(bad)
        duplicate_notes += sum(len(g) - len(set(g)) for g in bad)
        conflict_groups_total += len(groups)
    return {'rows_scanned': rows, 'conflict_windows': conflict_windows,
            'conflict_rate': conflict_windows / max(rows, 1),
            'duplicate_string_notes': duplicate_notes,
            'onset_groups': conflict_groups_total,
            'conflict_group_rate': duplicate_notes / max(conflict_groups_total, 1)}


def independent_loss(metadata, logits, labels, tuning, capo, conditional=False,
                     span_margin=4, stretch_margin=5):
    """Literal per-position formulas; deliberately avoids vector tricks."""
    import torch
    from open_fret.token_metadata import MAX_STRINGS
    ce_terms = []
    pitch_terms = []
    baseline_terms = []
    position_terms = []
    stretch_terms = []
    span_by_group = {}
    for row in range(labels.shape[0]):
        tune = [int(x) for x in tuning[row]]
        tick, note_order = 0, []
        for pos in range(labels.shape[1]):
            target = int(labels[row, pos])
            if target == -100:
                continue
            tok = metadata.tokens[target]
            if tok.startswith("TS_"):
                try:
                    tick += int(tok.split("_", 1)[1])
                except ValueError:
                    pass
                continue
            logp = logits[row, pos].float().log_softmax(-1)
            ce_terms.append(float((-logp[target]).detach()))
            note = metadata.note(target)
            if note is None:
                continue
            string, fret, _ = note
            if not 1 <= string <= MAX_STRINGS or fret > 24:
                continue
            pitch = int(tuning[row, string - 1]) + int(capo[row]) + fret
            equivalents = [i for i in range(len(metadata.tokens))
                           if metadata.pitch(i, tune, int(capo[row])) == pitch]
            pitch_terms.append(float((-torch.logsumexp(logp[equivalents], 0)).detach()))
            if conditional:
                playable = [i for i in range(len(metadata.tokens))
                            if metadata.pitch(i, tune, int(capo[row])) is not None]
                baseline_terms.append(float(torch.logsumexp(logp[playable], 0).detach()))
            exp_frets = {}
            for s in range(1, MAX_STRINGS + 1):
                if not 0 <= (tune[s - 1] if s - 1 < len(tune) else -1):
                    continue
                ids = [i for i in range(len(metadata.tokens))
                       if metadata.note(i) is not None and metadata.note(i)[0] == s
                       and 0 <= metadata.note(i)[1] <= 24]
                if not ids:
                    continue
                weights = torch.softmax(logp[ids], 0)
                exp_frets[s] = float((weights * torch.tensor(
                    [metadata.note(i)[1] for i in ids])).sum())
            if fret > 0:
                position_terms.append(abs(exp_frets.get(string, 0.0) - fret))
            note_order.append((tick, string, fret, exp_frets))
            key = (row, tick)
            span_by_group.setdefault(key, []).append((string, fret, exp_frets))
    for (row, tick), members in span_by_group.items():
        fretted = [(s, f, e) for s, f, e in members if f > 0 and s in e]
        if len(fretted) < 2:
            continue
        span = max(e[s] for s, _, e in fretted) - min(e[s] for s, _, e in fretted)
        span_by_group[(row, tick)] = max(0.0, span - span_margin)
    span_vals = [v for v in span_by_group.values() if isinstance(v, float)]
    for idx, (tick, string, fret, exp) in enumerate(note_order):
        if fret <= 0:
            continue
        window = [f for _, _, f, _ in note_order[max(0, idx - 5):idx + 6] if f > 0]
        if not window:
            continue
        stretch_terms.append(abs(exp.get(string, 0.0) - sum(window) / len(window)))
    stretch_vals = [max(0.0, v - stretch_margin) for v in stretch_terms]
    notes = max(len(pitch_terms), 1)
    return {'ce': sum(ce_terms) / max(len(ce_terms), 1),
            'pitch_marginal': sum(pitch_terms) / notes,
            'pitch_baseline': sum(baseline_terms) / notes,
            'pitch': (sum(pitch_terms) + sum(baseline_terms)) / notes,
            'position': sum(position_terms) / max(len(position_terms), 1),
            'span': sum(span_vals) / max(len(span_vals), 1),
            'n_span_groups': len(span_vals),
            'stretch': sum(stretch_vals) / max(len(position_terms), 1),
            'tokens': len(ce_terms), 'notes': len(pitch_terms)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--conflict-rows', type=int, default=20000)
    p.add_argument('--batch-rows', type=int, default=4)
    p.add_argument('--skip-scan', action='store_true')
    args = p.parse_args()
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    metadata = TokenMetadata.from_tokenizer(tok, 24)
    ds = load_from_disk(args.cache)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'cache': args.cache, 'test_music_read': False, 'splits': {}}
    if not args.skip_scan:
        for split in ('train', 'validation'):
            report['splits'][split] = fast_label_stats(ds[split], metadata, tok.eos_token_id)
            report['splits'][split]['conflicts'] = conflict_stats(
                ds[split], metadata, args.conflict_rows)
            print(split, report['splits'][split], flush=True)
            out.write_text(json.dumps(report, indent=2))

    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True)
    model.eval()
    batch = ds['validation'][:args.batch_rows]
    inputs = {k: torch.tensor(v) for k, v in batch.items()
              if k in ('input_ids', 'attention_mask', 'labels')}
    tuning = torch.tensor(batch['loss_tuning'])
    capo = torch.tensor(batch['loss_capo'])
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = logits.detach().requires_grad_(True)
    config = LossConfig(ce_weight=0.3, pitch_alpha=1.0, smoothing=0.1,
                        type_alpha=0.3, conditional_pitch=True,
                        position_alpha=0.5, span_alpha=0.5, span_margin=4,
                        stretch_alpha=0.5, stretch_margin=5)
    objective = TabLoss(metadata, config, eos_token_id=tok.eos_token_id)
    loss, parts = objective(logits, inputs['labels'], inputs['input_ids'],
                            loss_tuning=tuning, loss_capo=capo)
    reference = independent_loss(metadata, logits, inputs['labels'], tuning, capo,
                                 conditional=config.conditional_pitch)
    report['independent_check'] = {
        'rows': args.batch_rows, 'loss': float(loss),
        'ce': float(parts['ce']), 'reference': reference,
        'ce_abs_error': abs(float(parts['ce']) - reference['ce']),
        'pitch_abs_error': abs(float(parts['pitch']) - reference['pitch']),
        'position_abs_error': abs(float(parts['position']) - reference['position']),
        'span_abs_error': abs(float(parts['span']) - reference['span']),
        'stretch_abs_error': abs(float(parts['stretch']) - reference['stretch']),
        'weighted_identity_error': abs(float(loss) - (
            config.ce_weight * (float(parts['ce']) + float(parts['smoothing_delta']))
            + config.pitch_alpha * float(parts['pitch'])
            + config.type_alpha * float(parts['type'])
            + config.position_alpha * float(parts['position'])
            + config.span_alpha * float(parts['span'])
            + config.stretch_alpha * float(parts['stretch']))),
    }
    loss.backward()
    report['independent_check']['finite_gradients'] = bool(torch.isfinite(logits.grad).all())
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['independent_check'], indent=2))


if __name__ == '__main__':
    main()
