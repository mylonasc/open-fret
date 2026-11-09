import json
from pathlib import Path
import sys

import pretty_midi
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_fret.events import notes_to_tokens, midi_token_notes, validate_fingerings
from open_fret.midi_io import read_midi_notes
from open_fret.postprocess import NeighborSearchCorrector, CorrectionError
from open_fret.render import TabRenderer
from test_render_roundtrip import imported_attacks
from test_generation import setup  # real random tiny T5/tokenizer fixture
from open_fret.inference import transcribe_notes


def test_initial_silence_and_overlapping_equal_pitch_identity(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=480)
    inst = pretty_midi.Instrument(24)
    inst.notes = [pretty_midi.Note(90, 64, .5, 2), pretty_midi.Note(90, 64, 1, 1.5)]
    pm.instruments.append(inst)
    path = tmp_path / 'equal.mid'
    pm.write(str(path))
    notes, meta = read_midi_notes(str(path))
    assert [(n['start'], n['end']) for n in notes] == [(480, 1440), (960, 1920)]
    assert len({n['id'] for n in notes}) == 2
    assert notes_to_tokens(notes)[0] == 'TS_480'
    # MIDI has no note identity on release. Use FIFO rather than closing both
    # equal-pitch attacks at the first release (pretty_midi's behavior).
    assert [n['end'] for n in midi_token_notes(notes_to_tokens(notes))] == [1440, 1920]
    assert meta['equal_pitch_release_policy'].startswith('FIFO')
    assert meta['instrument'] == 0


def test_timed_matching_and_global_occupancy_backtracking():
    source = [{'id': 'a', 'start': 120, 'end': 960, 'pitch': 64},
              {'id': 'b', 'start': 480, 'end': 720, 'pitch': 59}]
    # At tick 120 pitch 64 can use strings 1 or 2; the later pitch needs 2.
    corr = NeighborSearchCorrector(tuning=[64, 59], max_fret=5)
    tokens, report = corr.correct('TAB_1_0 TS_120 TAB_2_5 TS_360 TAB_2_0',
                                  notes_to_tokens(source), source_notes=source)
    assert report.dropped_extra == 1  # wrong-tick note is not reused
    assert report.refingered == 1
    assert [n['string'] for n in report.corrected_notes] == [1, 2]
    assert tokens[0] == 'TS_120'
    validate_fingerings(report.corrected_notes, [64, 59], max_fret=5)


def test_impossible_sustain_and_search_cap_fail_without_partial_output():
    source = [{'id': 'a', 'start': 0, 'end': 960, 'pitch': 64},
              {'id': 'b', 'start': 480, 'end': 720, 'pitch': 64}]
    with pytest.raises(CorrectionError, match='sustained-string') as caught:
        NeighborSearchCorrector(tuning=[64]).correct([], notes_to_tokens(source), source_notes=source)
    assert caught.value.report.unresolved
    assert caught.value.report.corrected_notes == []
    with pytest.raises(CorrectionError, match='budget'):
        NeighborSearchCorrector(max_states=0).correct([], notes_to_tokens(source), source_notes=source)


def test_tuning_capo_range_and_release_before_attack():
    source = [{'id': 'a', 'start': 0, 'end': 480, 'pitch': 66},
              {'id': 'b', 'start': 480, 'end': 960, 'pitch': 66}]
    _, report = NeighborSearchCorrector(tuning=[64], capo=2).correct(
        ['TAB_99_99', 'TS_480', 'TAB_1_0'], notes_to_tokens(source), source_notes=source)
    assert [(n['string'], n['fret']) for n in report.corrected_notes] == [(1, 0), (1, 0)]
    assert report.pitch_fixed == 1
    with pytest.raises(CorrectionError, match='No legal fingering'):
        NeighborSearchCorrector(tuning=[40], max_fret=5).correct(
            [], notes_to_tokens(source), source_notes=source)


def test_independent_sustains_and_rests_render_exactly():
    notes = [{'id': 'a', 'start': 240, 'end': 2640, 'pitch': 64, 'string': 1, 'fret': 0},
             {'id': 'b', 'start': 480, 'end': 720, 'pitch': 59, 'string': 2, 'fret': 0},
             {'id': 'c', 'start': 960, 'end': 1320, 'pitch': 59, 'string': 2, 'fret': 0},
             {'id': 'd', 'start': 3000, 'end': 3180, 'pitch': 55, 'string': 3, 'fret': 0}]
    actual = imported_attacks(TabRenderer().to_alphatex_notes(notes))
    assert sorted((n['tick'], n['duration'], n['pitch']) for n in actual) == sorted(
        (n['start'], n['end'] - n['start'], n['pitch']) for n in notes)


def test_complete_song_chunking_preserves_ticks_and_sustains(setup):
    tok, model = setup
    notes = [{'id': str(i), 'start': 120 + i * 240, 'end': 240 + i * 240, 'pitch': 64}
             for i in range(12)]
    first = transcribe_notes(model, tok, notes, tuning=[64, 59], max_input_tokens=10, max_new_tokens=9)
    second = transcribe_notes(model, tok, notes, tuning=[64, 59], max_input_tokens=10, max_new_tokens=9)
    assert len(first['windows']) > 1
    assert first['raw_tokens'] == second['raw_tokens']
    assert first['raw_metrics']['timed_pitch']['f1'] == 1
    assert first['corrected_metrics']['timed_pitch']['f1'] == 1
    actual = first['correction']['corrected_notes']
    assert [(n['id'], n['start'], n['end'], n['pitch']) for n in actual] == [
        (n['id'], n['start'], n['end'], n['pitch']) for n in notes]


def test_silence_multi_instrument_selection_and_drum_policy(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    path = tmp_path / 'silence.mid'
    pm.write(str(path))
    assert read_midi_notes(str(path))[0] == []
    for i in range(2):
        inst = pretty_midi.Instrument(24)
        inst.notes.append(pretty_midi.Note(100, 64 + i, 0, .5))
        pm.instruments.append(inst)
    drum = pretty_midi.Instrument(0, is_drum=True)
    drum.notes.append(pretty_midi.Note(100, 40, 0, .5))
    pm.instruments.append(drum)
    pm.write(str(path))
    with pytest.raises(ValueError, match='Multiple pitched'):
        read_midi_notes(str(path))
    notes, meta = read_midi_notes(str(path), instrument=1)
    assert len(notes) == 1 and notes[0]['pitch'] == 65
    assert meta['ignored_instruments'] == [0]
    assert meta['drum_policy'] == 'channel 10 ignored'
