import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from open_fret.alphatex import AlphaTexParser, AlphaTexParseError, parse_tuning


def test_tuning_convention_matches_upstream():
    # Upstream _parse_tuning("E5") == 64 (standard high-e concert MIDI).
    assert parse_tuning("E5 B4 G4 D4 A3 E3") == [64, 59, 55, 50, 45, 40]
    assert parse_tuning("E3") == [40]


def test_single_notes_quarter_durations():
    parser = AlphaTexParser()
    content = "\\tempo 120\n\\track\n\\tuning E5 B4 G4 D4 A3 E3\n.\n:4 0.1 0.2 0.6\n"
    tracks = parser.parse_text(content)
    assert len(tracks) == 1
    durations = [e.duration for e in tracks[0].events]
    assert durations == [480, 480, 480], f"got {durations}"


def test_explicit_duration_override():
    parser = AlphaTexParser()
    content = "\\tempo 120\n\\track\n\\tuning E5 B4 G4 D4 A3 E3\n.\n:4 0.1.8 2.1\n"
    tracks = parser.parse_text(content)
    durations = [e.duration for e in tracks[0].events]
    assert durations == [240, 240], f"got {durations}"


def _events(body):
    return AlphaTexParser().parse_text("\\track\n\\tuning E5 B4 G4 D4 A3 E3\n.\n" + body)[0].events


def test_inline_chord_rest_and_directive_duration_inheritance():
    events = _events(":4 0.1.8 0.2 (0.1 2.2).16 0.3 r.8 0.4 :4 0.5")
    assert [e.duration for e in events] == [240, 240, 120, 120, 240, 480]
    assert [e.start for e in events] == [0, 240, 480, 600, 960, 1200]


def test_dotted_notes_chords_and_rests():
    events = _events(":4 0.1. 0.2{d} (0.1 2.2).8. 0.3 r.8{dd} 0.4")
    assert [e.duration for e in events] == [720, 720, 360, 240, 240]
    assert [e.start for e in events] == [0, 720, 1440, 1800, 2460]


@pytest.mark.parametrize("body", [
    ":4 0.1garbage 0.2", ":4 0.1{tu 3} 0.2", ":4 (0.1 garbage)",
    ":4 0.1{t} 0.2", ":4 0.1*3", ":3 0.1", ":4 0.7",
    ":4 (0.1.8 0.2)", "\\tempo 80\n:4 0.1", ":4 0.1{b (0 4)}",
    ":4 0.1\n\\tuning E5 B4 G4 D4 A3 E3\n0.2",
    "\\instrument 24 :4 0.1",
])
def test_unsupported_or_malformed_syntax_fails_visibly(body):
    with pytest.raises(ValueError):
        _events(body)


def test_bad_tuning_and_missing_files_fail_visibly(tmp_path):
    with pytest.raises(AlphaTexParseError):
        parse_tuning("E5 nonsense E3")
    with pytest.raises(FileNotFoundError):
        AlphaTexParser().parse_file(str(tmp_path / "missing.tex"))


def test_track_title_is_not_scanned_for_notes():
    tracks = AlphaTexParser().parse_text('\\track "Version 0.1 r"\n:4 0.2')
    assert len(tracks[0].events) == 1
    assert tracks[0].events[0].notes[0].string == 2


def test_technique_suffix_preserved():
    parser = AlphaTexParser()
    content = "\\tempo 120\n\\track\n\\tuning E5 B4 G4 D4 A3 E3\n.\n:4 2.3b4 5.6s 0.4v\n"
    tracks = parser.parse_text(content)
    techs = [e.notes[0].tech for e in tracks[0].events]
    assert techs == ["b4", "s", "v"], f"got {techs}"
    # Techniques must not corrupt durations.
    assert [e.duration for e in tracks[0].events] == [480, 480, 480]


def test_chord_and_rest():
    parser = AlphaTexParser()
    content = "\\tempo 120\n\\track\n\\tuning E5 B4 G4 D4 A3 E3\n.\n:4 (0.1 2.2) r.4 3.3\n"
    tracks = parser.parse_text(content)
    assert len(tracks[0].events) == 2, f"got {tracks[0].events}"
    assert sorted([(n.fret, n.string) for n in tracks[0].events[0].notes]) == [(0, 1), (2, 2)]
