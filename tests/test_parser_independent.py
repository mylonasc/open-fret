"""Optional independent cross-check: ALPHATAB_JS=/path/to/alphaTab.js pytest ..."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_fret.alphatex import AlphaTexParser


@pytest.mark.skipif(not os.environ.get("ALPHATAB_JS") or not shutil.which("node"),
                    reason="Set ALPHATAB_JS to independently verify with AlphaTab 1.3.0")
def test_inherited_durations_and_dots_against_alphatab():
    body = ":4 0.1.8 0.2 (0.1 2.2).16 0.3 r.8 0.4{d} :4 0.5"
    tex = '\\title "Timing probe"\n\\tempo 120\n.\n\\track\n\\tuning E5 B4 G4 D4 A3 E3\n' + body
    own = AlphaTexParser().parse_text(tex)[0]
    script = """
const at = require(process.argv[1]);
const score = at.importer.ScoreLoader.loadScoreFromBytes(
  new TextEncoder().encode(process.argv[2]), new at.Settings());
console.log(JSON.stringify(score.tracks[0].staves[0].bars[0].voices[0].beats
  .filter(b => !b.isRest).map(b => ({
    start: b.playbackStart / 2, duration: b.playbackDuration / 2,
    pitches: b.notes.map(n => n.realValue - 12).sort((a,b) => a-b)
  }))));
"""
    # AlphaTab 1.3 uses 960 PPQ vs our 480. Dataset octave names are one
    # octave above scientific notation: normalize its literal pitch by -12.
    imported = json.loads(subprocess.check_output(
        ["node", "-e", script, os.environ["ALPHATAB_JS"], tex], text=True))
    expected = [{"start": e.start, "duration": e.duration,
                 "pitches": sorted(own.tuning[n.string - 1] + n.fret for n in e.notes)}
                for e in own.events]
    assert imported == expected
