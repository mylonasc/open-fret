import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_fret.alphatab_parser import AlphaTabParser

pytestmark = pytest.mark.skipif(
    not shutil.which("node") or not (ROOT / "tools/alphatex_parser/node_modules").exists(),
    reason="npm ci in tools/alphatex_parser is required")


@pytest.fixture
def parser():
    worker = AlphaTabParser()
    yield worker
    worker.close()


def tex(body):
    return '\\title "Probe"\n\\tempo 120\n.\n\\track\n\\frets 24\n\\tuning E5 B4 G4 D4 A3 E3\n' + body


def test_pitch_strings_and_inherited_durations(parser):
    track = parser.parse_text(tex(":4 0.1 0.6.8 5.2"))[0]
    assert track.tuning == [64, 59, 55, 50, 45, 40]
    assert [(e.start, e.duration, e.notes[0].string) for e in track.events] == [
        (0, 480, 1), (480, 240, 6), (720, 240, 2)]


def test_tuplets_and_dots(parser):
    events = parser.parse_text(tex(":8{tu 3} 0.1 0.2 0.3 :4 0.1{d}"))[0].events
    assert [e.duration for e in events] == [160, 160, 160, 720]
    assert [e.start for e in events] == [0, 160, 320, 480]


def test_ties_not_rearticulated_and_common_effects_audited(parser):
    track = parser.parse_text(tex(":4 0.1 -.1 0.2{pm}"))[0]
    assert len(track.events) == 2
    assert track.events[0].duration == 960
    assert track.events[1].start == 960
    assert parser.diagnostics["articulations_not_targeted"]["isPalmMute"] == 1


def test_let_ring_duration_is_preserved(parser):
    events = parser.parse_text(tex(":4 0.1{lr} 0.2{lr} r"))[0].events
    assert events[0].duration > 480


def test_bad_syntax_rejected_without_poisoning_next_file(parser):
    with pytest.raises(ValueError):
        parser.parse_text(tex(":4 nonsense"))
    assert parser.parse_text(tex(":4 0.1"))[0].events[0].notes[0].fret == 0


def test_variable_tempo_is_explicitly_rejected(parser):
    with pytest.raises(ValueError, match="Variable tempo"):
        parser.parse_text(tex(":4 0.1 | \\tempo 90 :4 0.2"))
