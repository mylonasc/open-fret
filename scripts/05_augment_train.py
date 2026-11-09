"""Apply deterministic capo derivatives after family splitting, training only."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_fret.splits import assert_disjoint


def augment_train(data_dir, split_dir, output_root, seed=42):
    data_dir, split_dir, output = Path(data_dir), Path(split_dir), Path(output_root)
    if output.exists():
        raise FileExistsError('Augmentation output exists; use a new versioned directory')
    lists = {key: (split_dir / file).read_text().splitlines() for key, file in
             [('train', 'train_files.txt'), ('validation', 'val_files.txt'), ('test', 'test_files.txt')]}
    if any(not value for value in lists.values()):
        raise ValueError('Augmentation requires all three nonempty split lists')
    assignments, choices = {}, {}
    for split, names in lists.items():
        for name in names:
            row = json.loads((data_dir / name).read_text())
            if row.get('capo', 0) != 0:
                raise ValueError('Training augmentation expects unaugmented source windows')
            source = row.get('source') or row.get('file')
            if not source:
                raise ValueError('Missing source identity')
            assignments[name] = source
    assert_disjoint(*lists.values(), group_of=assignments.get)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.augment-', dir=output.parent) as tmp:
        root = Path(tmp) / 'data'
        individual = root / 'individual'
        individual.mkdir(parents=True)
        new_lists = {key: [] for key in lists}
        for split, names in lists.items():
            for name in names:
                row = json.loads((data_dir / name).read_text())
                (individual / name).write_text(json.dumps(row))
                new_lists[split].append(name)
                if split != 'train':
                    continue
                source = assignments[name]
                capo = choices.setdefault(source, random.Random(f'{seed}:{source}').randint(1, 7))
                derived = copy.deepcopy(row)
                derived['capo'] = capo
                derived['midi'] = [f'CAPO_{capo}' if t.startswith('CAPO_') else
                                   f'{t[:3]}{int(t[3:]) + capo}' if t.startswith(('NO_', 'NF_')) else t
                                   for t in row['midi']]
                if any(t.startswith(('NO_', 'NF_')) and int(t[3:]) > 127 for t in derived['midi']):
                    raise ValueError('Capo augmentation exceeds MIDI range')
                for key in ('notes', 'carry_notes'):
                    for note in derived.get(key, []):
                        note['pitch'] += capo
                        note['id'] = f"{note['id']}@capo{capo}"
                augmented_name = Path(name).stem + f'__aug_c{capo}.json'
                (individual / augmented_name).write_text(json.dumps(derived))
                new_lists[split].append(augmented_name)
        for split, filename in [('train', 'train_files.txt'), ('validation', 'val_files.txt'), ('test', 'test_files.txt')]:
            (root / filename).write_text('\n'.join(new_lists[split]) + '\n')
        manifest = json.loads((split_dir / 'split_manifest.json').read_text())
        manifest.update(augmentation_policy='post-split, deterministic capo augmentation training only',
                         augmentation_seed=seed, augmentation_choices=choices,
                         augmented_counts={key: len(new_lists[key]) for key in new_lists})
        (root / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
        os.rename(root, output)
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--split-dir', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    augment_train(args.data_dir, args.split_dir, args.output_root, args.seed)
