"""Restart-safe full pipeline. Existing legacy data is never reused by existence."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_fret.artifacts import run_stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', required=True, help='Isolated versioned data workspace; never an existing data root')
    ap.add_argument('--fixture', action='store_true', help='CPU synthetic end-to-end acceptance, no download')
    ap.add_argument('--run-name', help='Required for full training; never implicitly overwrites a run')
    ap.add_argument('--instrument-policy', choices=['mixed', 'guitar'], default='mixed')
    ap.add_argument('--prepare-only', action='store_true', help='Prepare/version data without starting training')
    args, training_args = ap.parse_known_args()
    if args.fixture:
        if args.prepare_only:
            ap.error('Fixture acceptance includes tiny training; --prepare-only is full-data only')
        if training_args:
            ap.error('Fixture mode does not accept production training arguments')
        subprocess.run([sys.executable, str(ROOT / 'scripts/fixture_pipeline.py'),
                        '--workspace', args.workspace], check=True, cwd=ROOT)
        return
    if not args.run_name and not args.prepare_only:
        ap.error('Full pipeline requires an explicit --run-name')
    workspace = Path(args.workspace).resolve()
    code = [ROOT / 'open_fret', ROOT / 'scripts', ROOT / 'requirements.txt',
            ROOT / 'tools/alphatex_parser/parse.cjs', ROOT / 'tools/alphatex_parser/package-lock.json']
    def stage(name, dependencies, command):
        return run_stage(workspace / name, dependencies + code, command, cwd=ROOT)
    raw = stage('raw', [], lambda out: [sys.executable, 'scripts/00_download.py', '--output-dir', str(out)])
    processed = stage('processed', [raw], lambda out: [sys.executable, 'scripts/01_process.py',
                      '--extracted-dir', str(raw / 'extracted'), '--processed-dir', str(out / 'individual'),
                      '--instrument-policy', args.instrument_policy])
    splits = stage('splits', [processed], lambda out: [sys.executable, 'scripts/03_split.py',
                   '--input-dir', str(processed / 'individual'), '--output-dir', str(out)])
    augmented = stage('augmented', [processed, splits], lambda out: [sys.executable, 'scripts/05_augment_train.py',
                      '--data-dir', str(processed / 'individual'), '--split-dir', str(splits),
                      '--output-root', str(out / 'data')])
    processed, splits = augmented / 'data', augmented / 'data'
    vocab = stage('vocab', [processed, splits], lambda out: [sys.executable, 'scripts/01.5_generate_vocab.py',
                  '--input-dir', str(processed / 'individual'), '--output', str(out / 'tokens.json'),
                  '--train-list', str(splits / 'train_files.txt'), '--complete-fretboard',
                  '--structural-lists', ','.join(str(splits / p) for p in ('train_files.txt', 'val_files.txt', 'test_files.txt'))])
    tokenizer = stage('tokenizer', [vocab], lambda out: [sys.executable, 'scripts/02_tokenize.py',
                      '--vocab', str(vocab / 'tokens.json'), '--output-dir', str(out)])
    cache = stage('cache', [processed, splits, tokenizer], lambda out: [sys.executable, 'scripts/04_cache_dataset.py',
                  '--data-dir', str(processed / 'individual'), '--tokenizer-dir', str(tokenizer),
                  '--train-list', str(splits / 'train_files.txt'), '--val-list', str(splits / 'val_files.txt'),
                  '--test-list', str(splits / 'test_files.txt'), '--cache-dir', str(out / 'dataset')])
    if args.prepare_only:
        print(f'Prepared cache: {cache / "dataset"}; tokenizer: {tokenizer}; no training started')
        return
    subprocess.run([sys.executable, 'scripts/train_model.py', '--run-name', args.run_name,
                    '--tokenizer-dir', str(tokenizer), '--cache-dir', str(cache / 'dataset'),
                    '--data-dir', str(processed / 'individual'), '--val-files', str(splits / 'val_files.txt'),
                    *training_args], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
