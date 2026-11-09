"""Musical held-out evaluation; validation selects, test is final reporting only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_fret.evaluation import evaluate_row, row_profile, summarize_rows


def evaluation_sources(dataset_path, split, num_samples, data_dir=None):
    from datasets import load_from_disk
    if split not in ('validation', 'test'):
        raise ValueError('Evaluation requires validation or test; no training fallback')
    if num_samples < 1:
        raise ValueError('num_samples must be positive')
    root = Path(dataset_path)
    dataset = load_from_disk(str(root))
    if split not in dataset:
        raise ValueError(f"Split {split!r} absent; refusing training fallback")
    if not len(dataset[split]):
        raise ValueError(f'Empty evaluation split: {split}')
    manifest = json.loads((root / 'cache_manifest.json').read_text())
    index = root / f'{split}.sources.jsonl'
    if not index.is_file():
        raise ValueError('Cache lacks unmasked source provenance; rebuild in a new strict-cache-v2 directory')
    directory = Path(data_dir or manifest.get('data_dir') or '')
    entries = []
    with index.open() as stream:
        for position, line in enumerate(stream):
            if position >= min(num_samples, len(dataset[split])):
                break
            identity = json.loads(line)
            content = (directory / identity['filename']).read_bytes()
            if hashlib.sha256(content).hexdigest() != identity['sha256']:
                raise ValueError('Evaluation source changed since cache construction')
            entries.append((identity['filename'], json.loads(content)))
    if len(entries) != min(num_samples, len(dataset[split])):
        raise ValueError('Partial source-provenance index')
    return entries, manifest


def evaluate_model(model_path, dataset_path, split='test', num_samples=100,
                   max_new_tokens=512, report_path=None, *, data_dir=None, modes=('legacy', 'anchored'),
                   tolerance=30, beams=4):
    entries, manifest = evaluation_sources(dataset_path, split, num_samples, data_dir)
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from open_fret.artifacts import tree_hash
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        raise RuntimeError(f'Could not load model/tokenizer from {model_path}: {e}') from e
    if torch.cuda.is_available():
        model = model.cuda()
    report = {'schema_version': 'musical-evaluation-v2', 'split': split, 'model': str(model_path),
              'model_identity': tree_hash(model_path) if Path(model_path).is_dir() else getattr(model.config, '_commit_hash', None),
              'tokenizer_sha256': hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
              'cache_manifest': manifest, 'samples': len(entries), 'rows': [],
              'generation': {'modes': list(modes), 'beams': beams, 'max_new_tokens': max_new_tokens,
                             'deterministic': True, 'legacy_no_repeat_ngram': 0, 'tolerance_ticks': tolerance},
              'use_policy': 'validation for selection; test once after selection is frozen'}
    for name, data in entries:
        report['rows'].append({'file': name, **row_profile(data, name),
                              'generations': evaluate_row(model, tokenizer, data, modes=modes,
                                  beams=beams, max_new_tokens=max_new_tokens, tolerance=tolerance)})
    report['summary'] = summarize_rows(report['rows'])
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(target)
    print(json.dumps(report['summary'], indent=2))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', choices=['validation', 'test'], default='test')
    ap.add_argument('--num-samples', type=int, default=100)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--report', required=True)
    ap.add_argument('--data-dir')
    ap.add_argument('--beams', type=int, default=4)
    ap.add_argument('--tolerance-ticks', type=int, default=30)
    args = ap.parse_args()
    evaluate_model(args.model, args.dataset, args.split, args.num_samples,
                    args.max_new_tokens, args.report, data_dir=args.data_dir,
                    beams=args.beams, tolerance=args.tolerance_ticks)
