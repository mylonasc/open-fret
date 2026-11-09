# Hosted viewer demo (GitHub Pages)

Live page: <https://mylonasc.github.io/open-fret/examples/player.html>

The page proves that the interactive viewer works as a static site: AlphaTab
notation and playback, a MIDI piano roll, click-to-seek, zoom, and the
predicted/actual toggle all run in the browser. No Python, GPU, or model
inference is involved after the build.

## What is published

| File | Content |
| --- | --- |
| `examples/player.html` | Viewer bundle inlined, synthetic example data |
| `examples/example.mid` | Source MIDI for the demo phrase |
| `examples/predicted.tex` | First fingering (AlphaTex) |
| `examples/reference.tex` | Pitch-equivalent alternative fingering |
| `licenses.html` | Project license plus bundled and runtime components |
| `demo-manifest.json` | Provenance, note count, and publication flags |

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

`.github/workflows/pages-demo.yml` builds the page on Ubuntu, uploads it as a
Pages artifact, and deploys with `actions/deploy-pages`. The workflow installs
only the small I/O dependencies (`numpy`, `pretty_midi`, `mido`), so it never
downloads a corpus or model.

Pages is configured with `build_type: workflow`. The `github-pages` environment
allows the `main` and `first-working-training-loop` branches; remove the latter
once the work is merged to `main`.

## What this does not establish

The demo is not a model-quality claim and does not clear the rights to publish
trained weights. See [dataset licensing](dataset_licensing.md) before releasing
any corpus-derived examples or checkpoints.
