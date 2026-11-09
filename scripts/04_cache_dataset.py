"""Strict, atomic aligned-window cache; cached/uncached rows share one encoder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from open_fret.alphatex import SUPPORTED_PARSER_VERSIONS, StaleParserDataError
from open_fret.curation import conflict_positions
from open_fret.encoding import encode_window


def create_cache(tokenizer_dir="dataset/processed/tab_tokenizer",
                 data_dir="dataset/processed/individual",
                 train_list="dataset/processed/train_files.txt",
                 val_list="dataset/processed/val_files.txt",
                 test_list="dataset/processed/test_files.txt",
                 cache_dir="dataset/processed/cached_dataset",
                 max_length=512, mask_conflicts=True, require_test=False):
    destination = Path(cache_dir)
    if destination.exists():
        raise FileExistsError(f"Cache destination exists: {destination}; use a new versioned directory")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    counts = {"masked_windows": 0, "masked_notes": 0, "scanned_windows": 0}
    curation = {"enabled": mask_conflicts, "totals": counts.copy(), "splits": {}}
    parser_version, split_hashes = None, {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cache-build-', dir=destination.parent) as workspace:
        splits = {}
        for split, file_list in (("train", train_list), ("validation", val_list), ("test", test_list)):
            path = Path(file_list)
            if not path.exists() and split == 'test' and not require_test:
                continue  # explicit development caller policy, never train fallback
            names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            if not names:
                raise ValueError(f"Empty {split} split: {path}")
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate rows in {split} list")
            stats = curation['splits'][split] = counts.copy()
            def gen():
                nonlocal parser_version
                digest = hashlib.sha256()
                for name in names:
                    source = Path(data_dir) / name
                    try:
                        content = source.read_bytes()
                        data = json.loads(content)
                        version = data.get('parser_version')
                        if version not in SUPPORTED_PARSER_VERSIONS:
                            raise StaleParserDataError(f"Obsolete parser_version in {name}; reprocess in a new directory")
                        if parser_version is None:
                            parser_version = version
                        if parser_version != version:
                            raise StaleParserDataError("Mixed parser versions; rebuild in an empty directory")
                        encoded = encode_window(data, tokenizer, max_length, mask_conflicts)
                    except Exception as e:
                        raise ValueError(f"Invalid cache source {source}: {e}") from e
                    digest.update(name.encode() + b'\0' + content + b'\0')
                    # Row order + source identity is auditable without making
                    # strings/sidecars part of the model's tensor inputs.
                    identity = {key: data.get(key) for key in
                                ('source', 'track', 'window', 'start_tick', 'end_tick', 'window_version')}
                    identity.update(filename=name, sha256=hashlib.sha256(content).hexdigest())
                    with (Path(workspace) / f'{split}.sources.jsonl').open('a', encoding='utf-8') as provenance:
                        provenance.write(json.dumps(identity) + '\n')
                    if mask_conflicts:
                        ignored = conflict_positions(data['tab'])
                        for target in (stats, curation['totals']):
                            target['scanned_windows'] += 1
                            target['masked_windows'] += bool(ignored)
                            target['masked_notes'] += len(ignored)
                    yield encoded
                split_hashes[split] = digest.hexdigest()

            splits[split] = Dataset.from_generator(gen, cache_dir=str(Path(workspace) / split))
        output = Path(workspace) / 'complete'
        DatasetDict(splits).save_to_disk(str(output))
        for split in splits:
            os.rename(Path(workspace) / f'{split}.sources.jsonl', output / f'{split}.sources.jsonl')
        manifest = {"schema_version": "strict-cache-v2", "status": "complete",
                    "tokenizer_dir": str(tokenizer_dir), "max_length": max_length,
                    "tokenizer_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
                    "source_sha256": split_hashes, "parser_version": parser_version,
                    "mask_conflicts": mask_conflicts, "curation": curation,
                    "skipped_or_truncated": 0, "splits": {k: len(v) for k, v in splits.items()}}
        (output / 'cache_manifest.json').write_text(json.dumps(manifest, indent=2))
        os.rename(output, destination)
    print(f"Cache complete: {destination} ({manifest['splits']}); no skipped/truncated rows")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer-dir', default='dataset/processed/tab_tokenizer')
    ap.add_argument('--data-dir', default='dataset/processed/individual')
    ap.add_argument('--cache-dir', default='dataset/processed/cached_dataset')
    ap.add_argument('--train-list', default='dataset/processed/train_files.txt')
    ap.add_argument('--val-list', default='dataset/processed/val_files.txt')
    ap.add_argument('--test-list', default='dataset/processed/test_files.txt')
    ap.add_argument('--max-length', type=int, default=512)
    ap.add_argument('--allow-no-test', action='store_true', help='Development-only cache; missing test is explicit')
    ap.add_argument('--no-mask-conflicts', action='store_true')
    a = ap.parse_args()
    create_cache(tokenizer_dir=a.tokenizer_dir, data_dir=a.data_dir, cache_dir=a.cache_dir,
                 train_list=a.train_list, val_list=a.val_list, test_list=a.test_list,
                 max_length=a.max_length, mask_conflicts=not a.no_mask_conflicts,
                 require_test=not a.allow_no_test)
