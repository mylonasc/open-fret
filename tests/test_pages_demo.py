import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_pages_demo import build, synthetic_example
from open_fret.events import validate_fingerings


def test_pages_demo_is_synthetic_and_contains_both_players(tmp_path):
    output = tmp_path / 'site'
    manifest = build(output)
    page = (output / 'examples/player.html').read_text()
    assert 'alphaTab.min.js' in page
    assert 'wave-roll' in page
    assert 'html-midi-player' in page
    assert 'Choose predicted or actual tab' in page
    assert 'piano roll' in page
    assert 'not a model prediction' in page
    assert (output / 'examples/example.mid').read_bytes().startswith(b'MThd')
    assert (output / 'examples/predicted.tex').read_text() != (output / 'examples/reference.tex').read_text()
    assert manifest['notes'] == 36
    assert manifest['piano_roll'] == 'WaveRoll 0.4.0'
    assert not manifest['weights_published'] and not manifest['corpus_examples_published']
    assert json.loads((output / 'demo-manifest.json').read_text())['status'] == 'complete'
    assert 'Facebook, Inc.' in (output / 'licenses.html').read_text()
    assert {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == {
        'index.html', 'licenses.html', 'demo-manifest.json', '.nojekyll',
        'examples/player.html', 'examples/example.mid', 'examples/predicted.tex', 'examples/reference.tex'}
    with pytest.raises(FileExistsError):
        build(output)


def test_both_synthetic_fingerings_preserve_all_pitches_and_sustains():
    source, predicted, reference = synthetic_example()
    tuning = [64, 59, 55, 50, 45, 40]
    validate_fingerings(predicted, tuning)
    validate_fingerings(reference, tuning)
    assert [{k: n[k] for k in ('id', 'start', 'end', 'pitch')} for n in predicted] == source
    assert [{k: n[k] for k in ('id', 'start', 'end', 'pitch')} for n in reference] == source
    assert any(a['string'] != b['string'] for a, b in zip(predicted, reference))
