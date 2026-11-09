import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.viewer import (  # noqa: E402
    build_view_bundle,
    midi_tokens_to_notes,
    notes_to_midi_bytes,
    to_html,
)


def test_midi_tokens_to_notes_pairs_on_off():
    notes = midi_tokens_to_notes("NO_60 TS_480 NF_60 NO_64 TS_240 NF_64".split())
    assert notes == [
        {"start": 0, "end": 480, "pitch": 60},
        {"start": 480, "end": 720, "pitch": 64},
    ]


def test_midi_tokens_stray_off_and_dangling_on():
    notes = midi_tokens_to_notes("NF_60 NO_62 TS_100".split())
    # Stray OFF ignored; dangling ON extends by default_duration.
    assert notes == [{"start": 0, "end": 480, "pitch": 62}]


def test_midi_bytes_roundtrip():
    notes = midi_tokens_to_notes("NO_60 TS_480 NF_60 NO_67 TS_480 NF_67".split())
    data = notes_to_midi_bytes(notes, tempo=120)
    assert data[:4] == b"MThd"
    import pretty_midi
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        pm = pretty_midi.PrettyMIDI(path)
        pitches = sorted(n.pitch for inst in pm.instruments for n in inst.notes)
        assert pitches == [60, 67], pitches
    finally:
        os.unlink(path)


def _sample_window():
    # Redistributable, self-contained fixture; never read a git-ignored corpus.
    return {'midi': 'CAPO_0 TUNING_64_59_55_50_45_40 NO_64 NO_59 TS_240 NF_64 NF_59 TS_120 NO_67 TS_480 NF_67'.split(),
            'tab': 'TAB_1_0 TAB_2_0 TS_360 TAB_1_3 TS_480'.split()}


def test_build_view_bundle_from_tokens():
    d = _sample_window()
    bundle = build_view_bundle(midi_tokens=d["midi"], tab_tokens=d["tab"],
                               title="test")
    assert bundle["midi_b64"]
    raw = base64.b64decode(bundle["midi_b64"])
    assert raw[:4] == b"MThd"
    assert "\\tempo" in bundle["alpha_tex"] and "\\track" in bundle["alpha_tex"]
    assert bundle["meta"]["n_input_notes"] > 0
    assert bundle["input_notes"] == midi_tokens_to_notes(d["midi"])
    assert bundle["tab_tokens"] == d["tab"]
    assert "reference_alpha_tex" not in bundle


def test_build_view_bundle_includes_reference_tab():
    d = _sample_window()
    bundle = build_view_bundle(midi_tokens=d["midi"], tab_tokens=d["tab"],
                               reference_tab_tokens=d["tab"], title="test")
    assert bundle["reference_tab_tokens"] == d["tab"]
    assert bundle["reference_alpha_tex"] != bundle["alpha_tex"]
    assert "(actual reference)" in bundle["reference_alpha_tex"]
    assert bundle["meta"]["n_reference_tab_tokens"] == len(d["tab"])


def test_to_html_contains_mount_points():
    bundle = {"title": "t", "midi_b64": "AAA=", "alpha_tex": "x",
              "input_tokens": [], "tab_tokens": [],
              "meta": {}, "tempo": 120, "ticks_per_beat": 480}
    html = to_html(bundle, dist_js="/* fake bundle */")
    assert "ofv-" in html
    assert "alphaTab" in html and "wave-roll" in html
    assert "midi-player" not in html
    assert "/* fake bundle */" in html
    assert "AAA=" in html
