import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_fret.evaluation import (freeze_validation_panel, sequence_metrics, evaluate_row,
                                  row_profile, summarize_rows)
from open_fret.transcription_diagnostics import compare_streams
from test_generation import setup
from test_loss_cli import fixture_cache, load_script


def test_missing_extra_shifted_and_empty_outputs_reduce_scores():
    midi = 'CAPO_0 TUNING_64_59 NO_64 TS_120 NF_64 NO_59 TS_120 NF_59'.split()
    ref = 'TAB_1_0 TS_120 TAB_2_0 TS_120'.split()
    assert compare_streams(midi, ref, ref)['timed_pitch']['f1'] == 1
    for bad in ([], ref[:2], ref + ['TAB_1_0'], ['TS_60'] + ref):
        assert compare_streams(midi, ref, bad)['timed_pitch']['f1'] < 1
        assert sequence_metrics(ref, bad)['edit_distance'] > 0
    assert compare_streams(midi, ref, ['TAB_broken'])['prediction']['invalid_notes'] == 1


def test_panel_is_frozen_stratified_and_never_filters_unsupported_sources(tmp_path):
    data = tmp_path / 'individual'
    data.mkdir()
    names = []
    for index, tokens in enumerate(('NO_64 TS_120 NF_64', 'NO_64 NO_59 TS_120 NF_64 NF_59', 'NO_127 TS_120 NF_127')):
        name = f'Song{index}__t0__c0__w0.json'
        names.append(name)
        (data / name).write_text(json.dumps({'source': f'Song{index}.tex',
            'midi': ['CAPO_0', 'TUNING_64_59'] + tokens.split(),
            'tab': ['TAB_1_0', 'TS_120'], 'instrument_domain': ['guitar', 'bass', 'unknown'][index]}))
    (tmp_path / 'val_files.txt').write_text('\n'.join(names))
    (tmp_path / 'test_files.txt').write_text('this-file-must-not-be-read.json')
    panel = freeze_validation_panel(tmp_path, tmp_path / 'panel.json', 3)
    assert len(panel['rows']) == 3
    assert panel['test_music_read'] is False
    assert {r['file'] for r in panel['rows']} == set(names)
    assert 'domain:bass' in panel['available_strata']
    assert freeze_validation_panel(tmp_path, tmp_path / 'panel.json', 3) == panel
    (data / names[0]).write_text('{}')
    with pytest.raises(ValueError, match='source changed'):
        freeze_validation_panel(tmp_path, tmp_path / 'panel.json', 3)


def test_failed_generations_stay_in_source_family_denominators(setup):
    tok, model = setup
    good = {'midi': 'CAPO_0 TUNING_64_59 NO_64 TS_120 NF_64'.split(),
            'tab': 'TAB_1_0 TS_120'.split(), 'source': 'Good.tex'}
    bad = {**good, 'midi': 'CAPO_0 TUNING_64_59 NO_127 TS_120 NF_127'.split(), 'source': 'Unsupported.tex'}
    rows = [{**row_profile(d, d['source']), 'generations': evaluate_row(model, tok, d, modes=('anchored',))}
            for d in (good, bad)]
    summary = summarize_rows(rows)
    anchored = summary['modes']['anchored']
    assert anchored['attempted_sources'] == 2
    assert anchored['failed_sources'] == 1
    assert anchored['macro_f1_failures_as_zero'] == .5
    assert anchored['macro_f1_success_only'] == 1
    assert anchored['family_cluster_bootstrap_95_percent'] is not None
    assert rows[0]['generations']['anchored']['timed_pitch']['tolerance_seconds'] == .03125
    assert 'baseline' in rows[0]['generations']
    assert 'anchored_corrected' in rows[0]['generations']


def test_heldout_cache_requires_real_split_and_original_unmasked_sources(fixture_cache, tmp_path):
    evaluate = load_script('evaluate.py')
    with pytest.raises(ValueError, match='absent'):
        evaluate.evaluation_sources(fixture_cache / 'cache', 'test', 100)
    with pytest.raises(ValueError, match='no training fallback'):
        evaluate.evaluation_sources(fixture_cache / 'cache', 'train', 100)
    entries, _ = evaluate.evaluation_sources(fixture_cache / 'cache', 'validation', 100)
    assert len(entries) == 2  # small datasets are not indexed past their end
    copy = tmp_path / 'sources'
    shutil.copytree(fixture_cache / 'individual', copy)
    (copy / entries[0][0]).write_text('{}')
    with pytest.raises(ValueError, match='source changed'):
        evaluate.evaluation_sources(fixture_cache / 'cache', 'validation', 100, data_dir=copy)


def test_small_real_model_evaluation_emits_versioned_report(fixture_cache, tmp_path):
    from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(fixture_cache / 'tokenizer')
    model = T5ForConditionalGeneration(T5Config(vocab_size=len(tok), d_model=16, d_ff=32,
        num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.pad_token_id))
    model.save_pretrained(tmp_path / 'model')
    tok.save_pretrained(tmp_path / 'model')
    report = load_script('evaluate.py').evaluate_model(str(tmp_path / 'model'),
        str(fixture_cache / 'cache'), split='validation', num_samples=100,
        report_path=tmp_path / 'report.json', modes=('anchored',))
    assert report['samples'] == 2
    assert report['model_identity']
    assert report['tokenizer_sha256']
    assert report['summary']['modes']['anchored']['macro_f1_failures_as_zero'] == 1
    assert json.loads((tmp_path / 'report.json').read_text())['schema_version'] == 'musical-evaluation-v2'
