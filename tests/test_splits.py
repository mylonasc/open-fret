import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.splits import split_by_group, assert_disjoint, group_id_from_filename
from open_fret.splits import canonical_family, musical_fingerprint
import json
import importlib.util
import pytest


def test_group_split_no_leakage():
    files = [f"song{i//2}__item_{i%2}.json" for i in range(20)]
    groups = [group_id_from_filename(f) for f in files]
    train, val, test = split_by_group(files, groups, seed=0)
    assert_disjoint(train, val, test)
    # All derivatives of one song stay together.
    for song in {f.split('__item_')[0] for f in files}:
        parts = [f for f in files if f.startswith(song + '__item_')]
        assert len({('train' if p in train else 'val' if p in val else 'test') for p in parts}) == 1


def test_version_families_and_fingerprints_keep_timing_and_multiplicity():
    assert canonical_family('Aerosmith_-_Dream_On.tex') == canonical_family('collection/Aerosmith_-_Dream_On_(3).tex')
    def rows(tokens):
        return [{'midi': tokens.split(), 'capo': 0, 'start_tick': 0, 'track': 0}]
    original = rows('NO_64 TS_480 NF_64')
    assert musical_fingerprint(original) != musical_fingerprint(rows('NO_64 TS_240 NF_64'))
    assert musical_fingerprint(original) != musical_fingerprint(rows('NO_64 NO_64 TS_480 NF_64 NF_64'))


def test_family_splits_exclude_eval_derivatives_and_are_reproducible(tmp_path):
    spec = importlib.util.spec_from_file_location('fixture_split', Path(__file__).resolve().parents[1] / 'scripts/03_split.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / 'data'
    data.mkdir()
    for source in list(range(10)) + ['0_(2)']:
        pitch = 64 + (int(str(source).split('_')[0]) % 10)
        for capo in (0, 3):
            row = dict(source=f'Artist_Song{source}.tex', track=0, window=0, capo=capo,
                       midi=[f'NO_{pitch + capo}', 'TS_480', f'NF_{pitch + capo}'], tab=['TAB_1_0', 'TS_480'])
            (data / f'{source}__c{capo}.json').write_text(json.dumps(row))
    first = module.split_dataset(str(data), str(tmp_path / 'first'))
    second = module.split_dataset(str(data), str(tmp_path / 'second'))
    assert first == second
    assert first['groups']['Artist_Song0.tex'] == first['groups']['Artist_Song0_(2).tex']
    assert first['excluded_eval_derivatives'] > 0
    for name in ('val_files.txt', 'test_files.txt'):
        assert all('__c0' in f for f in (tmp_path / 'first' / name).read_text().splitlines())
    with pytest.raises(FileExistsError):
        module.split_dataset(str(data), str(tmp_path / 'first'))


def test_new_scheme_grouping():
    files = [
        "Band__Song__t0__c0__w0.json",
        "Band__Song__t0__c0__w1.json",
        "Band__Song__t1__c3__w0.json",
        "Other__Track__t0__c0__w0.json",
    ]
    groups = [group_id_from_filename(f) for f in files]
    assert groups[:3] == ["Band__Song"] * 3
    assert groups[3] == "Other__Track"
    train, val, test = split_by_group(files, groups, train_ratio=0.5,
                                      val_ratio=0.25, seed=0)
    assert_disjoint(train, val, test)
