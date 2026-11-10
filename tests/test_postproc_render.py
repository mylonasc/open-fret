import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.postprocess import NeighborSearchCorrector, notes_to_tab_tokens, relocate_deviating_notes
from open_fret.render import TabRenderer


def _note(start, end, pitch, string, fret, id_=None):
    return {"id": id_ or f"{start}-{pitch}", "start": start, "end": end,
            "pitch": pitch, "string": string, "fret": fret}


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


def test_relocate_prefers_open_string_for_outlier():
    tuning = [64, 59, 55, 50, 45, 40]
    notes = [
        _note(0, 480, 66, 1, 2), _note(480, 960, 67, 1, 3),
        _note(960, 1440, 64, 4, 14),
        _note(1440, 1920, 67, 1, 3), _note(1920, 2400, 66, 1, 2),
    ]
    moved, report = relocate_deviating_notes(notes, tuning=tuning)
    assert report.relocated == 1
    assert [(n["string"], n["fret"]) for n in moved][2] == (1, 0)
    assert [n["pitch"] for n in moved] == [n["pitch"] for n in notes]
    assert [n["pitch"] for n in moved] == [66, 67, 64, 67, 66]


def test_relocate_picks_string_closest_to_neighbourhood_mean():
    tuning = [64, 59, 55, 50, 45, 40]
    notes = [
        _note(0, 480, 66, 1, 2), _note(480, 960, 67, 1, 3),
        _note(960, 1440, 76, 3, 21),
        _note(1440, 1920, 67, 1, 3), _note(1920, 2400, 66, 1, 2),
    ]
    moved, report = relocate_deviating_notes(notes, tuning=tuning)
    assert report.relocated == 1
    # Pitch 76 has no open-string option (64+12, 59+17, 55+21); s1 f12 wins.
    assert [(n["string"], n["fret"]) for n in moved][2] == (1, 12)
    assert [n["pitch"] for n in moved] == [n["pitch"] for n in notes]


def test_relocate_leaves_playable_line_alone():
    notes = [_note(i * 480, (i + 1) * 480, 64 + [0, 2, 3, 2, 0][i], 1, [0, 2, 3, 2, 0][i])
             for i in range(5)]
    moved, report = relocate_deviating_notes(notes)
    assert report.relocated == 0
    assert [(n["string"], n["fret"]) for n in moved] == [(1, f) for f in (0, 2, 3, 2, 0)]


def test_relocate_respects_max_fret_and_busy_strings():
    # Boxed-in outlier: fretted neighbours establish a low hand position,
    # but every alternative string is occupied for the outlier's duration,
    # so the note is left untouched (still pitch-exact).
    notes = [
        _note(0, 480, 66, 1, 2), _note(480, 960, 67, 1, 3),
        _note(960, 1440, 64, 1, 0), _note(960, 1440, 59, 2, 0),
        _note(960, 1440, 55, 3, 0), _note(960, 1440, 70, 4, 20),
        _note(1440, 1920, 67, 1, 3), _note(1920, 2400, 66, 1, 2),
        _note(2400, 2880, 67, 1, 3), _note(2880, 3360, 66, 1, 2),
        _note(3360, 3840, 67, 1, 3),
    ]
    moved, report = relocate_deviating_notes(notes)
    # The boxed-in outlier itself cannot move (every alternative string is
    # occupied for its duration); neighbours may still shift toward the
    # local mean, exactly as the reference algorithm behaves.
    assert [(n["string"], n["fret"]) for n in moved][5] == (4, 20)
    assert [n["pitch"] for n in moved] == [n["pitch"] for n in notes]
    for n in moved:
        assert 0 <= n["fret"] <= 24
    # A note above a tighter max_fret is relocated into range instead.
    solo = [_note(0, 480, 64, 4, 14)]
    moved2, report2 = relocate_deviating_notes(solo, max_fret=10)
    assert report2.relocated == 1 and report2.out_of_range == 1
    assert moved2[0]["fret"] <= 10
    assert moved2[0]["pitch"] == 64


def test_notes_to_tab_tokens_roundtrip():
    notes = [_note(0, 480, 64, 1, 0), _note(480, 960, 67, 1, 3),
             _note(480, 960, 55, 3, 0)]
    assert notes_to_tab_tokens(notes) == (
        ["TAB_1_0", "TS_480", "TAB_1_3", "TAB_3_0", "TS_480"])
