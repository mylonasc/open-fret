import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_fret.domains import accept_instrument, instrument_domain
from open_fret.alphatex import ParsedTrack, ParsedNote, TimedEvent
from test_loss_cli import load_script


def test_declared_instrument_policy_never_calls_bass_guitar():
    assert instrument_domain(24, 6) == 'guitar'
    assert instrument_domain(33, 4) == 'bass'
    assert instrument_domain(None, 6) == 'other_or_unknown'
    assert accept_instrument(33, 4, 'mixed')
    assert not accept_instrument(33, 4, 'guitar')
    assert not accept_instrument(None, 6, 'guitar')
    assert not accept_instrument(24, 13, 'mixed')


def test_source_sampler_equalizes_source_mass_and_replays_by_epoch():
    from open_fret.sampling import SourceBalancedSampler
    sources = ['large'] * 90 + ['small'] * 10
    sampler = SourceBalancedSampler(sources, seed=42)
    assert abs(float(sampler.weights[:90].sum() - sampler.weights[90:].sum())) < 1e-10
    first = list(sampler)
    assert list(sampler) == first
    assert len(first) == len(sources)
    sampler.set_epoch(1)
    assert list(sampler) != first
    resumed = SourceBalancedSampler(sources, seed=42)
    resumed.set_epoch(1)
    assert list(resumed) == list(sampler)


def test_augmentation_is_post_split_train_only_and_reproducible(tmp_path):
    data, splits = tmp_path / 'data', tmp_path / 'splits'
    data.mkdir()
    splits.mkdir()
    for index, split in enumerate(('train', 'validation', 'test')):
        name = split + '.json'
        note = {'id': f'{split}:0', 'start': 240, 'end': 720, 'pitch': 64, 'string': 1, 'fret': 0}
        (data / name).write_text(json.dumps({'source': f'{split}.tex', 'capo': 0,
             'midi': ['CAPO_0', 'TUNING_64_59', 'TS_240', 'NO_64', 'TS_480', 'NF_64'],
             'tab': ['TS_240', 'TAB_1_0', 'TS_480'], 'notes': [note], 'carry_notes': []}))
        filename = {'train': 'train_files.txt', 'validation': 'val_files.txt', 'test': 'test_files.txt'}[split]
        (splits / filename).write_text(name + '\n')
    (splits / 'split_manifest.json').write_text('{}')
    augment = load_script('05_augment_train.py').augment_train
    first = augment(data, splits, tmp_path / 'first')
    second = augment(data, splits, tmp_path / 'second')
    assert first == second
    assert first['augmented_counts'] == {'train': 2, 'validation': 1, 'test': 1}
    capo = first['augmentation_choices']['train.tex']
    derivative = json.loads(next((tmp_path / 'first/individual').glob('*__aug_*.json')).read_text())
    assert derivative['notes'][0]['pitch'] == 64 + capo
    assert derivative['notes'][0]['start'] == 240 and derivative['notes'][0]['end'] == 720
    assert f'NO_{64+capo}' in derivative['midi']
    assert derivative['tab'] == ['TS_240', 'TAB_1_0', 'TS_480']
    for split in ('validation', 'test'):
        assert json.loads((tmp_path / 'first/individual' / f'{split}.json').read_text())['capo'] == 0
    with pytest.raises(FileExistsError):
        augment(data, splits, tmp_path / 'first')


def test_source_rejection_publishes_no_partial_windows(tmp_path, monkeypatch):
    process = load_script('01_process.py')
    good = ParsedTrack([64, 59], 120, [TimedEvent(0, 480, [ParsedNote(0, 1)])])
    bad = ParsedTrack([64, 59], 120, [TimedEvent(0, 480, [ParsedNote(-1, 1)])])
    class Parser:
        def parse_file(self, path):
            return [good, bad]
    monkeypatch.setattr(process, 'AlphaTexParser', Parser)
    output = tmp_path / 'output'
    output.mkdir()
    count, error, _ = process._process_one((str(tmp_path / 'song.tex'), str(output), str(tmp_path), 42, 'strict', 'mixed'))
    assert count == 0 and 'Invalid source fret' in error
    assert not list(output.iterdir())
