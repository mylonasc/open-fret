"""Content-addressed completion checks for restartable pipeline stages."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def tree_hash(path):
    path = Path(path)
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"Empty stage artifact: {path}")
    for file in files:
        if file.name == '_stage_complete.json' or '__pycache__' in file.parts:
            continue
        digest.update(str(file.relative_to(path) if path.is_dir() else file.name).encode() + b'\0')
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def run_stage(destination, inputs, make_command, *, cwd=None, env=None):
    destination = Path(destination)
    identity = {str(p): tree_hash(p) for p in inputs}
    # Commands normalize the temporary staging path before identity comparison.
    command_id = make_command(Path('<OUTPUT>'))
    expected = {'version': 1, 'inputs': identity, 'command': command_id}
    if destination.exists():
        manifest = destination / '_stage_complete.json'
        if not manifest.is_file():
            raise ValueError(f"Partial/unversioned stage {destination}; choose a new workspace")
        saved = json.loads(manifest.read_text())
        if any(saved.get(k) != v for k, v in expected.items()) or saved.get('output_hash') != tree_hash(destination):
            raise ValueError(f"Stale/corrupt stage {destination}; choose a new workspace")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stage-', dir=destination.parent) as tmp:
        output = Path(tmp) / 'output'
        output.mkdir()
        subprocess.run(make_command(output), check=True, cwd=cwd, env=env)
        expected.update(status='complete', output_hash=tree_hash(output))
        (output / '_stage_complete.json').write_text(json.dumps(expected, indent=2))
        os.rename(output, destination)
    return destination
