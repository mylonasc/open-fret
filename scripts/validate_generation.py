"""Seeded source-balanced validation audit; never reads held-out test music."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_fret.generation import AnchoredTemplate, generate_tab, UnsupportedInputError
from open_fret.splits import group_id_from_filename
from open_fret.transcription_diagnostics import compare_streams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sources', type=int, default=128)
    p.add_argument('--decode-n', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    root = Path(args.data_root)
    groups = {}
    for name in (root / 'val_files.txt').read_text().splitlines():
        if '__c0__' in name:
            groups.setdefault(group_id_from_filename(name), []).append(name)
    rng = random.Random(args.seed)
    names = [rng.choice(sorted(groups[g])) for g in rng.sample(sorted(groups), min(args.sources, len(groups)))]
    report = {'checkpoint': args.checkpoint, 'seed': args.seed, 'test_music_read': False,
              'source_balanced': True, 'capo': 0, 'rows': []}
    for n, name in enumerate(names):
        d = json.loads((root / 'individual' / name).read_text())
        row = {'file': name, 'strings': len(d['tuning']),
               'technique_targets': sum(t.startswith('TAB_') and len(t.split('_')) > 3 for t in d['tab'])}
        try:
            template = AnchoredTemplate(tok, d['midi'])
            ref_ids = [tok.get_vocab().get(t, -1) for t in d['tab']] + [tok.eos_token_id]
            prefix = [model.config.decoder_start_token_id]
            errors = Counter()
            if len(ref_ids) != len(template.slots):
                errors['slot_count'] += 1
            else:
                for i, target in enumerate(ref_ids):
                    if target not in template.slots[i].ids:
                        errors['pitch_or_structure'] += 1
                    if target not in template.allowed(i, prefix):
                        errors['chord_constraint'] += 1
                    prefix.append(target)
            row['reference_violations'] = dict(errors)
            if n < args.decode_n:
                row['generations'] = {}
                for mode in ('legacy', 'anchored'):
                    result = generate_tab(model, tok, d['midi'], mode=mode)
                    row['generations'][mode] = {**result.metadata,
                        **compare_streams(d['midi'], d['tab'], result.tokens),
                        'exact_reference': result.tokens == d['tab']}
                print(n, name, 'anchored F1', row['generations']['anchored']['timed_pitch']['f1'], flush=True)
        except (UnsupportedInputError, RuntimeError) as e:
            row['error'] = str(e)
        report['rows'].append(row)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(out)


if __name__ == '__main__':
    main()
