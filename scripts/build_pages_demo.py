"""Build a static, synthetic-only AlphaTab/piano-roll proof for GitHub Pages.

No dataset, checkpoint, training, inference service or credentials are read.
The two hand-authored fingerings demonstrate the viewer, not model quality.
"""
import argparse
import base64
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_fret.events import notes_to_tokens
from open_fret.viewer import build_view_bundle, renderer_for_input, to_html


def synthetic_example():
    pitches = [64, 67, 69, 71, 74, 72, 69, 67]
    gaps = [240, 240, 480, 240, 240, 480, 240, 720]
    source, predicted, reference = [], [], []
    tick = 240
    for i in range(32):
        pitch, gap = pitches[i % len(pitches)], gaps[i % len(gaps)]
        chord = i in (8, 24)
        notes = [(pitch, tick + gap - 60)]
        if chord:
            notes += [(55, tick + gap + 480), (59, tick + gap - 60)]
        for p, end in sorted(notes):
            note = {'id': f'demo-{len(source)}', 'start': tick, 'end': end, 'pitch': p}
            source.append(note)
            string, fret = ((3, 0) if p == 55 else (2, 0) if p == 59 else (1, p - 64))
            predicted.append({**note, 'string': string, 'fret': fret})
            alt = (2, p - 59) if p >= 64 and not chord else (string, fret)
            reference.append({**note, 'string': alt[0], 'fret': alt[1]})
        tick += gap
    return source, predicted, reference


def tab_tokens(notes):
    tokens, tick = [], 0
    times = sorted({n[key] for n in notes for key in ('start', 'end')})
    for time in times:
        if time > tick:
            tokens.append(f'TS_{time - tick}')
        tick = time
        for note in notes:
            if note['start'] == time:
                tokens.append(f"TAB_{note['string']}_{note['fret']}")
    return tokens


def build(output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Output exists: {output}; use a new build directory')
    source, predicted, reference = synthetic_example()
    midi = ['CAPO_0', 'TUNING_64_59_55_50_45_40'] + notes_to_tokens(source)
    bundle = build_view_bundle(midi_tokens=midi, tab_tokens=tab_tokens(predicted),
                               reference_tab_tokens=tab_tokens(reference),
                               title='Synthetic guitar sketch', tempo=100)
    renderer = renderer_for_input(midi, tempo=100)
    bundle['alpha_tex'] = renderer.to_alphatex_notes(predicted, title='Synthetic sketch — fingering A')
    bundle['reference_alpha_tex'] = renderer.to_alphatex_notes(reference, title='Synthetic sketch — fingering B')
    bundle['expected_notes'] = source
    bundle['provenance'] = 'hand-authored synthetic fixture; no dataset or trained model'
    dist = (ROOT / 'ui/tab_data_viewer/dist/tab-data-viewer.umd.js').read_text()
    viewer = to_html(bundle, dist_js=dist)
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="A live static AlphaTab score and MIDI piano roll on GitHub Pages.">
<link rel="icon" href="data:,">
<title>Open-Fret · Interactive viewer proof</title>
<style>
body { margin:0; background:#f1f5f9; color:#182236; font:16px/1.6 system-ui,sans-serif; }
main { max-width:1400px; margin:auto; padding:32px 20px; }
h1 { font-size:clamp(25px,4vw,42px); line-height:1.2; margin:8px 0 16px; }
.eyebrow { color:#4f46e5; font-weight:700; letter-spacing:.1em; font-size:12px; }
.intro { max-width:900px; color:#475569; }
.notice { border-left:4px solid #6366f1; padding:12px 16px; background:#eef2ff; border-radius:6px; }
.viewer { background:white; padding:20px; margin:24px 0; border:1px solid #dbe2ec; border-radius:16px; }
a { color:#4338ca; } nav { display:flex; gap:16px; flex-wrap:wrap; margin:16px 0; }
footer { color:#64748b; font-size:14px; } .ofv { max-width:none; }
@media(max-width:600px) { main {padding:24px 10px;} .viewer {padding:10px;} }
</style></head><body><main>
<div class="eyebrow">OPEN-FRET / GITHUB PAGES PROOF</div>
<h1>Guitar tablature meets the piano roll.</h1>
<p class="intro">A static HTML page with interactive JavaScript: source MIDI on the left,
AlphaTab notation on the right. No Python server, GPU or model runs in your browser.</p>
<p class="notice"><strong>Try it:</strong> click <strong>▶ MIDI input</strong> or <strong>▶ Tab</strong>.
Switch <strong>Predicted / Actual</strong> to see two pitch-equivalent fingerings;
click the score to seek, or change the zoom. The players are independent, not sample-synchronized.</p>
<div class="viewer">''' + viewer + '''</div>
<h2>What this example proves</h2>
<p>GitHub Pages can serve both the piano roll and playable AlphaTab, including the
fingering toggle and seek controls. Audio starts after a user gesture; scripts,
notation fonts and soundfonts are loaded from external CDNs.</p>
<p><strong>Synthetic demonstration, not a model prediction.</strong> This small fixture
was created for Open-Fret. The “Predicted” and “Actual” panels contain hand-authored
alternative fingerings solely to exercise the existing viewer. No training songs,
reference annotations from a corpus, or model weights are published here.</p>
<nav aria-label="Downloads and project links">
<a href="example.mid" download>Download MIDI</a>
<a href="predicted.tex" download>Fingering A · AlphaTex</a>
<a href="reference.tex" download>Fingering B · AlphaTex</a>
<a href="https://github.com/mylonasc/open-fret">Source repository</a>
<a href="../licenses.html">Licenses &amp; third-party notices</a>
</nav>
<footer>Original Open-Fret code and this synthetic fixture: MIT.
React is bundled; AlphaTab and the MIDI/audio components retain their own licenses.</footer>
</main></body></html>
'''
    examples = output / 'examples'
    examples.mkdir(parents=True)
    (examples / 'player.html').write_text(page, encoding='utf-8')
    (examples / 'example.mid').write_bytes(base64.b64decode(bundle['midi_b64']))
    (examples / 'predicted.tex').write_text(bundle['alpha_tex'], encoding='utf-8')
    (examples / 'reference.tex').write_text(bundle['reference_alpha_tex'], encoding='utf-8')
    (output / '.nojekyll').touch()
    (output / 'index.html').write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0;url=./examples/player.html"><title>Open-Fret viewer</title>
</head><body><a href="./examples/player.html">Open the interactive AlphaTab and piano-roll example</a></body></html>''')
    own = (ROOT / 'LICENSE').read_text()
    third_party = (ROOT / 'ui/tab_data_viewer/THIRD_PARTY_NOTICES.md').read_text()
    (output / 'licenses.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Open-Fret · Licenses</title><h1>Licenses and notices</h1><pre style="white-space:pre-wrap">'
        + html.escape(own + '\n\n' + third_party) + '</pre></html>', encoding='utf-8')
    manifest = {'status': 'complete', 'provenance': bundle['provenance'], 'notes': len(source),
                'tempo': 100, 'expected_notes': source, 'piano_roll': 'WaveRoll 0.4.0',
                'weights_published': False,
                'corpus_examples_published': False}
    (output / 'demo-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', required=True)
    result = build(ap.parse_args().output_dir)
    print(f"Built synthetic Pages example: {result['notes']} notes; no weights/corpus assets")
