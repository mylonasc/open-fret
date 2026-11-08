import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.alphatex import ParsedTrack, ParsedNote, TimedEvent
from open_fret.sequences import SequenceBuilder, AlignedWindower


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
