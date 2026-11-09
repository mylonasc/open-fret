"""Fail-fast dataset download and atomic, traversal-safe extraction."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import zipfile

DATASET_NAME = 'vldsavelyev/guitar_tab'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'dataset/raw'


def extract_archive(archive, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Extraction destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.extract-', dir=destination.parent) as tmp:
        root = Path(tmp).resolve()
        with zipfile.ZipFile(archive) as source:
            bad = source.testzip()
            if bad:
                raise ValueError(f"Corrupt archive member: {bad}")
            for member in source.infolist():
                path = (root / member.filename).resolve()
                if not path.is_relative_to(root) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            source.extractall(root)
        if not any(root.rglob('*.tex')):
            raise ValueError("Archive contains no AlphaTex files")
        (root / '_extraction_manifest.json').write_text(json.dumps({'schema_version': 1, 'status': 'complete'}))
        os.rename(root, destination)


def download_and_save_dataset(output_dir=OUTPUT_DIR):
    from huggingface_hub import hf_hub_download
    output_dir = Path(output_dir)
    archive = hf_hub_download(repo_id=DATASET_NAME, filename='data.zip',
                              repo_type='dataset', local_dir=str(output_dir))
    extract_archive(archive, output_dir / 'extracted')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=str(OUTPUT_DIR))
    download_and_save_dataset(ap.parse_args().output_dir)
