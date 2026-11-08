import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_fret.render import TabRenderer, duration_pieces
from open_fret.transcription_diagnostics import event_stream

PACKAGE = ROOT / "tools/alphatex_parser/node_modules/@coderline/alphatab"
pytestmark = pytest.mark.skipif(not shutil.which("node") or not PACKAGE.exists(),
                                reason="npm ci --prefix tools/alphatex_parser required")


def imported_attacks(tex):
    program = """
const at=require(process.argv[1]), fs=require('fs');
const settings=new at.Settings();
const s=at.importer.ScoreLoader.loadScoreFromBytes(new TextEncoder().encode(fs.readFileSync(0,'utf8')),settings);
const gen=new at.midi.MidiFileGenerator(s,settings,{}), events=[];
for(const track of s.tracks) for(const staff of track.staves) for(const bar of staff.bars)
 for(const voice of bar.voices) for(const beat of voice.beats) for(const note of beat.notes)
  if(!note.isTieDestination) {
    const d=gen.getNoteDuration(note,beat.playbackDuration);
    events.push({tick:beat.absolutePlaybackStart/2,pitch:note.realValue,
                 duration:Math.max(d.untilTieOrSlideEnd,d.letRingEnd)/2});
  }
console.log(JSON.stringify(events));
"""
    return json.loads(subprocess.check_output(["node", "-e", program, str(PACKAGE)], input=tex, text=True))


@pytest.mark.parametrize("tokens", [
    "TAB_1_0 TS_480 TAB_2_0 TS_240",
    "TAB_1_0 TS_180 TAB_2_0 TS_90 TAB_3_0 TS_360 TAB_4_0 TS_720 TAB_5_0 TS_1440",
    "TAB_1_0 TS_80 TAB_2_0 TS_160 TAB_3_0 TS_320 TAB_4_0 TS_640",
    "TS_240 TAB_1_0 TS_120 TS_360 TAB_2_0 TS_600",
    "TAB_1_0 TAB_2_0 TS_600 TAB_3_0 TS_2400 TAB_4_0 TS_480",
    "TAB_1_0 TS_10000 TAB_2_0 TS_240",
    "TAB_1_0 TS_117 TAB_2_0 TS_480",
])
def test_complete_exact_attack_roundtrip(tokens):
    tex = TabRenderer().to_alphatex(tokens)
    expected, _ = event_stream(tokens, "tab", [64, 59, 55, 50, 45, 40])
    actual = imported_attacks(tex)
    assert sorted((e["tick"], e["pitch"]) for e in actual) == sorted(
        (e["tick"], e["pitch"]) for e in expected)


def test_dotted_duration_value_and_beat_syntax():
    tex = TabRenderer().to_alphatex("TAB_1_0 TS_180 TAB_2_0 TS_720")
    actual = imported_attacks(tex)
    assert ":16 0.1{d}" in tex
    assert [e["duration"] for e in actual] == [180, 720]


def test_rest_only_and_unrepresentable_time_are_not_silently_lost():
    assert imported_attacks(TabRenderer().to_alphatex("TS_480 TS_240")) == []
    with pytest.raises(ValueError, match="represent exactly"):
        TabRenderer().to_alphatex("TAB_1_0 TS_1 TAB_2_0 TS_480")
    with pytest.raises(ValueError, match="Negative"):
        TabRenderer().to_alphatex("TAB_1_0 TS_-1")


def test_large_composite_duration_has_no_recursive_remainder_loop():
    assert sum(p.ticks for p in duration_pieces(1921)) == 1921
    assert sum(p.ticks for p in duration_pieces(10001)) == 10001
