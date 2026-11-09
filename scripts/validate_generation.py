"""Frozen source-balanced validation only; failed sources stay in denominators."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_fret.evaluation import freeze_validation_panel, evaluate_row, summarize_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--manifest', help='Immutable panel provenance; defaults to OUT.panel.json')
    ap.add_argument('--sources', type=int, default=512)
    ap.add_argument('--decode-n', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tolerance-ticks', type=int, default=30)
    args = ap.parse_args()
    if args.decode_n < 1 or args.tolerance_ticks < 0:
        ap.error('decode-n must be positive and tolerance nonnegative')
    root, out = Path(args.data_root), Path(args.out)
    # Freeze selection independently of model/decoder feasibility.
    panel = freeze_validation_panel(root, args.manifest or str(out) + '.panel.json', args.sources, args.seed)
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from open_fret.artifacts import tree_hash
    import torch
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint, local_files_only=True).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    report = {'schema_version': 'generated-validation-v2', 'checkpoint': args.checkpoint,
              'checkpoint_sha256': tree_hash(args.checkpoint),
              'tokenizer_sha256': hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
              'seed': args.seed, 'test_music_read': False, 'panel': panel,
              'decode_n': args.decode_n, 'tolerance_ticks': args.tolerance_ticks, 'rows': []}
    for index, entry in enumerate(panel['rows']):
        data = json.loads((root / 'individual' / entry['file']).read_text())
        row = dict(entry)
        if index < args.decode_n:
            row['generations'] = evaluate_row(model, tokenizer, data, tolerance=args.tolerance_ticks)
        else:
            row['not_decoded'] = 'outside declared decode budget; not filtered for feasibility'
        report['rows'].append(row)
        report['summary'] = summarize_rows(report['rows'], args.seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(out)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
