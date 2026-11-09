"""Versioned song-family/full-source-fingerprint splits; augmentation train-only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_fret.splits import split_by_group, assert_disjoint, canonical_family, musical_fingerprint


def split_dataset(input_dir, output_dir, train_ratio=.8, val_ratio=.1, seed=42):
    root, output = Path(input_dir), Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Processed input directory missing: {root}")
    if any((output / name).exists() for name in ('train_files.txt', 'val_files.txt', 'test_files.txt', 'split_manifest.json')):
        raise FileExistsError("Split artifacts exist; use a new versioned output directory")
    files = sorted(p for p in root.glob('*.json') if not p.name.startswith('_'))
    if not files:
        raise ValueError("No processed windows")
    rows, sources, digest = {}, {}, hashlib.sha256()
    for path in files:
        content = path.read_bytes()
        row = json.loads(content)
        if not isinstance(row.get('midi'), list) or not isinstance(row.get('tab'), list):
            raise ValueError(f"Invalid window schema: {path}")
        source = row.get('source') or row.get('file')
        if not source:
            raise ValueError(f"Missing source identity: {path}")
        # Keep only identities in memory, not an entire corpus of token arrays.
        rows[path.name] = {'source': source, 'capo': row.get('capo', 0),
                           'track': row.get('track', 0),
                           'instrument_domain': row.get('instrument_domain', 'unknown')}
        sources.setdefault(source, []).append(path.name)
        digest.update(path.name.encode() + b'\0' + content)
    parent = {s: s for s in sources}
    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    def union(a, b):
        a, b = sorted((find(a), find(b)))
        parent[b] = a
    named, exact, fingerprints = {}, {}, {}
    for source in sorted(sources):
        family = canonical_family(source)
        fingerprint = musical_fingerprint([json.loads((root / name).read_text()) for name in sources[source]])
        fingerprints[source] = fingerprint
        for key, table in ((family, named), (fingerprint, exact)):
            if key in table:
                union(source, table[key])
            else:
                table[key] = source
    names = list(rows)
    groups = [find(rows[n].get('source') or rows[n]['file']) for n in names]
    result = split_by_group(names, groups, train_ratio, val_ratio, seed)
    file_groups = dict(zip(names, groups))
    assert_disjoint(*result, group_of=file_groups.get)
    filtered = [result[0]] + [[n for n in split if rows[n].get('capo', 0) == 0] for split in result[1:]]
    if any(not split for split in filtered):
        raise ValueError("Empty split after family grouping; supply enough independent sources (>=10 for default ratios)")
    manifest = {'schema_version': 'family-split-v2', 'status': 'complete', 'seed': seed,
                'train_ratio': train_ratio, 'val_ratio': val_ratio, 'source_sha256': digest.hexdigest(),
                'augmentation_policy': 'derived capos only in train; primary validation/test capo=0',
                'near_duplicate_policy': 'no fuzzy/intro-riff merges; named versions and full-source exact fingerprints only',
                'groups': {s: find(s) for s in sorted(sources)}, 'fingerprints': fingerprints,
                'n_groups': len(set(groups)), 'sources': len(sources),
                'tracks': len({(r.get('source') or r['file'], r.get('track', 0)) for r in rows.values()}),
                'splits': {key: {'windows': len(split), 'sources': len({rows[n].get('source') or rows[n]['file'] for n in split}),
                                 'groups': len({file_groups[n] for n in split})}
                           for key, split in zip(('train', 'validation', 'test'), filtered)},
                'excluded_eval_derivatives': sum(len(s) for s in result) - sum(len(s) for s in filtered)}
    output.mkdir(parents=True, exist_ok=True)
    for filename, split in zip(('train_files.txt', 'val_files.txt', 'test_files.txt'), filtered):
        (output / filename).write_text('\n'.join(split) + '\n')
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='dataset/processed/individual')
    ap.add_argument('--output-dir', default='dataset/processed')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    split_dataset(args.input_dir, args.output_dir, seed=args.seed)
