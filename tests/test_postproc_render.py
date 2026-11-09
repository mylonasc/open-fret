import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.postprocess import NeighborSearchCorrector
from open_fret.render import TabRenderer


def test_missing_note_inserted():
    corr = NeighborSearchCorrector()
    out, report = corr.correct(["TAB_1_0"], ["NO_64", "NO_67"])
    assert len([t for t in out if t.startswith("TAB_")]) == 2, out
    assert report.inserted == 1


def test_same_string_chord_fixed():
    corr = NeighborSearchCorrector()
    out, report = corr.correct(["TAB_1_0", "TAB_1_3"], ["NO_64", "NO_67"])
    strings = [int(t.split("_")[1]) for t in out if t.startswith("TAB_")]
    assert len(strings) == 2 and strings[0] != strings[1], out


def test_renderer_first_note_has_duration():
    r = TabRenderer()
    tex = r.to_alphatex("TAB_1_0 TS_480 TAB_2_0 TS_240")
    assert ":4" in tex and ":8" in tex, tex
    # First note must carry a duration prefix, not dangle.
    assert tex.index(":4") < tex.index("0.1"), tex


def test_ascii_aligned_width():
    r = TabRenderer()
    ascii_tab = r.to_ascii("TAB_1_10 TS_480 TAB_2_3 TS_480")
    lines = ascii_tab.split("\n")
    assert len(lines) == 6
    widths = {len(line) for line in lines}
    assert len(widths) == 1, lines


def test_zero_shift_stays_one_chord_and_custom_string_count_is_not_lost():
    renderer = TabRenderer(tuning_str='E5 B4 G4 D4 A3 E3 B2')
    assert renderer.to_ascii('TAB_1_0 TS_0 TAB_7_10') == renderer.to_ascii('TAB_1_0 TAB_7_10')
    lines = renderer.to_ascii('TAB_1_0 TAB_7_10').splitlines()
    assert len(lines) == 7
    assert len({len(line) for line in lines}) == 1
    assert '10' in lines[-1]
