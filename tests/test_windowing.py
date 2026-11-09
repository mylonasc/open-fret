import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.alphatex import ParsedTrack, ParsedNote, TimedEvent
from open_fret.sequences import SequenceBuilder, AlignedWindower
import pytest
from open_fret.events import tab_attacks


def _track(n_onsets=10, gap=480):
    events = []
    for i in range(n_onsets):
        events.append(TimedEvent(start=i * gap, duration=240,
                                 notes=[ParsedNote(fret=i % 5, string=1)]))
    return ParsedTrack(tuning=[64, 59, 55, 50, 45, 40], tempo=120, events=events)


def test_aligned_windows_cover_same_span():
    track = _track(20)
    builder = SequenceBuilder()
    steps, tuning = builder.build_steps(track, capo=0)
    windower = AlignedWindower(max_input_tokens=40, max_target_tokens=40)
    windows = windower.window(steps, tuning, capo=0, source="s", track_index=0)
    assert len(windows) > 1
    for w in windows:
        assert len(w.midi_tokens) <= 40
        assert len(w.tab_tokens) <= 40
        # Every target TAB onset must have a corresponding MIDI onset.
        n_tab = sum(1 for t in w.tab_tokens if t.startswith("TAB_"))
        n_no = sum(1 for t in w.midi_tokens if t.startswith("NO_"))
        assert n_tab == n_no, f"{w.midi_tokens} vs {w.tab_tokens}"


def test_no_silent_truncation_eos_reserved():
    track = _track(200)
    builder = SequenceBuilder()
    steps, tuning = builder.build_steps(track, capo=0)
    windower = AlignedWindower(max_input_tokens=512, max_target_tokens=512)
    windows = windower.window(steps, tuning)
    total_ons = sum(sum(1 for t in w.tab_tokens if t.startswith("TAB_")) for w in windows)
    assert total_ons == 200


def test_absolute_onsets_identity_ends_and_carry_policy():
    track = ParsedTrack(tuning=[64, 59], tempo=120, events=[
        TimedEvent(start=240, duration=1920, notes=[ParsedNote(fret=0, string=1)]),
        TimedEvent(start=480, duration=120, notes=[ParsedNote(fret=0, string=2)]),
        TimedEvent(start=960, duration=240, notes=[ParsedNote(fret=2, string=2)]),
    ])
    steps, tuning = SequenceBuilder().build_steps(track)
    windows = AlignedWindower(9, 9).window(steps, tuning, source='song')
    assert len(windows) > 1
    attacks = [(w.start_tick + n['start'], n['string'], n['fret'])
               for w in windows for n in tab_attacks(w.tab_tokens)]
    assert attacks == [(240, 1, 0), (480, 2, 0), (960, 2, 2)]
    notes = [n for w in windows for n in w.notes]
    assert len({n['id'] for n in notes}) == len(notes) == 3
    assert [(n['start'], n['end']) for n in notes] == [(240, 2160), (480, 600), (960, 1200)]
    assert any(n['end'] == 2160 for w in windows[1:] for n in w.carry_notes)
    assert all(len(w.midi_tokens) + 1 <= 9 and len(w.tab_tokens) + 1 <= 9 for w in windows)


def test_oversized_single_onset_fails_instead_of_truncation():
    track = ParsedTrack(tuning=[64, 59], tempo=120, events=[
        TimedEvent(start=0, duration=480, notes=[ParsedNote(fret=0, string=1)] * 10)])
    steps, tuning = SequenceBuilder().build_steps(track)
    with pytest.raises(ValueError, match='Single timed step'):
        AlignedWindower(8, 8).window(steps, tuning)
