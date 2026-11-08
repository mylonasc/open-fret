import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.alphatex import AlphaTexParser


def test_single_notes_quarter_durations():
    parser = AlphaTexParser()
    content = "\\tempo 120\n\\track\n\\tuning E4 B3 G3 D3 A2 E2\n.\n:4 0.1 0.2 0.6\n"
    tracks = parser.parse_text(content)
    assert len(tracks) == 1
    durations = [e.duration for e in tracks[0].events]
    assert durations == [480, 480, 480], f"got {durations}"


def test_explicit_duration_override():
    parser = AlphaTexParser()
    content = "\\tempo 120\n\\track\n\\tuning E4 B3 G3 D3 A2 E2\n.\n:4 0.1.8 2.1\n"
    tracks = parser.parse_text(content)
    durations = [e.duration for e in tracks[0].events]
    assert durations == [240, 480], f"got {durations}"


def test_technique_suffix_preserved():
    parser = AlphaTexParser()
    content = "\\tempo 120\n\\track\n\\tuning E4 B3 G3 D3 A2 E2\n.\n:4 2.3b4 5.6s 0.4v\n"
    tracks = parser.parse_text(content)
    techs = [e.notes[0].tech for e in tracks[0].events]
    assert techs == ["b4", "s", "v"], f"got {techs}"
    # Techniques must not corrupt durations.
    assert [e.duration for e in tracks[0].events] == [480, 480, 480]


def test_chord_and_rest():
    parser = AlphaTexParser()
    content = "\\tempo 120\n\\track\n\\tuning E4 B3 G3 D3 A2 E2\n.\n:4 (0.1 2.2) r.4 3.3\n"
    tracks = parser.parse_text(content)
    assert len(tracks[0].events) == 2, f"got {tracks[0].events}"
    assert sorted([(n.fret, n.string) for n in tracks[0].events[0].notes]) == [(0, 1), (2, 2)]
