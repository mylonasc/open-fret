# Hosted synthetic demo (served from the documentation site)

Live page: <https://mylonasc.github.io/open-fret/examples/synthetic/>

The page proves that the interactive viewer works as a static site: AlphaTab
notation and playback, a MIDI piano roll, click-to-seek, zoom, and the
predicted/actual toggle all run in the browser. No Python, GPU, or model
inference is involved after the build. It is built by
`scripts/build_pages_demo.py` and deployed as part of the documentation site
(`website/static/examples/synthetic/`).

## What is published

| File (under `examples/synthetic/`) | Content |
| --- | --- |
| `index.html` | Viewer bundle inlined, synthetic example data |
| `example.mid` | Source MIDI for the demo phrase |
| `predicted.tex` | First fingering (AlphaTex) |
| `reference.tex` | Pitch-equivalent alternative fingering |

The piano roll is rendered by [WaveRoll](https://github.com/crescent-stdio/wave-roll)
0.4.0 (`createPianoRoll`, MIT). It supplies the pitch-range view, C-octave
labels, time grid, zoom/pan and a scrolling playhead. AlphaTab is the single
audio source and clock: its `tickPosition` drives the transport seek bar, the
tab cursor and the roll playhead together, so the views cannot drift apart.
A green playback bar tracks the sounding measures (site CSS AlphaTab needs
but does not ship). A Listen toggle flips the loaded score between the tab
and a pitched rendering of the exact input notes carried in the bundle
(`midi_alpha_tex`), so the fingering can be A/B-compared against the input
through the same player with the position preserved.
It replaced Magenta's `PianoRollSVGVisualizer` (previously used via
`html-midi-player`), which had no time or pitch axis and stayed blank until
playback began.
Alternatives considered: `@molecule/app-feature-piano-roll-react` needs
private Molecule peer packages; `react-piano-roll` and `webaudio-pianoroll`
are unmaintained editor widgets. Among the candidates reviewed, WaveRoll was
the only maintained, framework-agnostic roll with a documented
rendering-only factory.

Everything on the page is **hand-authored synthetic material**. The repository's
training corpus, reference annotations, checkpoints, and generated previews are
not uploaded.

## Build locally

```bash
npm ci --prefix ui/tab_data_viewer
npm --prefix ui/tab_data_viewer run build
python scripts/build_pages_demo.py --output-dir /tmp/open-fret-pages
```

Then serve `/tmp/open-fret-pages` over HTTP (for example
`python -m http.server --directory /tmp/open-fret-pages`) and open
`examples/player.html`. Opening the file directly with `file://` does not work
reliably because the page loads scripts and a soundfont from CDNs.

The build refuses to overwrite an existing output directory and writes a
manifest with `weights_published: false` and `corpus_examples_published: false`.
`tests/test_pages_demo.py` checks the artifact set, pitch preservation, valid
fingering, and the overwrite guard.

## Deployment

The documentation site (`.github/workflows/website.yml`) builds the
Docusaurus site on Ubuntu, copies the synthetic demo above into
`website/static/examples/synthetic/`, uploads `website/build` as a Pages
artifact, and deploys with `actions/deploy-pages`. The workflow installs only
the small I/O dependencies (`numpy`, `pretty_midi`, `mido`) for the synthetic
demo plus the website's `npm ci`, so it never downloads a corpus or model.
The old `examples/player.html` URL keeps working via a static redirect page.

Pages is configured with `build_type: workflow`. The `github-pages` environment
allows the `main` and `first-working-training-loop` branches; remove the latter
once the work is merged to `main`.

## What this does not establish

The demo is not a model-quality claim and does not clear the rights to publish
trained weights. See [dataset licensing](dataset_licensing.md) before releasing
any corpus-derived examples or checkpoints.
