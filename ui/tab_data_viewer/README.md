# tab_data_viewer

Playable + seekable React viewer for open-fret model I/O pairs (Issue #13).

- **Input:** MIDI notes (played via `html-midi-player` piano-roll).
- **Output:** predicted TAB tokens rendered by **AlphaTab** (notation + tab,
  audio playback, beat cursor, click-the-score to seek).
- Embeddable in a web page (`demo/index.html`) or a Jupyter cell via
  `open_fret.viewer.show()`.

## Confirmed CDN endpoints (checked 2026-11-08)

- AlphaTab: `https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/alphaTab.min.js`
- SoundFont: `.../dist/soundfont/sonivox.sf2`
- MIDI stack: jsDelivr `combine/` of `tone@14.7.77` +
  `@magenta/music@1.23.0/es6/core.js` + `focus-visible@5` +
  `html-midi-player@1.5.0` (note: `@magenta/music@1.23.2` does **not** exist;
  `html-midi-player@1.6.0` changes the dist layout, so 1.5.0 is pinned).

## Verified API facts (from upstream docs, not guesses)

- `html-midi-player`: `<midi-player src visualizer="#id" sound-font>`,
  `<midi-visualizer type="piano-roll" id>`, JS `.start()` / `.stop()`,
  `.currentTime`, `load`/`note`/`start`/`stop` events.
- AlphaTab UMD exposes `window.alphaTab.AlphaTabApi`; score clicks seek
  natively when the player is enabled. Everything else (mute/solo,
  `playbackSpeed`, tick position, beat events) is **feature-detected** in the
  component — missing APIs degrade to warnings, not crashes.

## Build

```bash
cd ui/tab_data_viewer
npm install
npm run build   # -> dist/tab-data-viewer.umd.js (React bundled in)
```

Serve the demo (ES modules/file URLs are fine for the demo page since the
bundle is UMD, but `fetch()` of the JSON needs http):

```bash
npx vite preview --outDir demo  # or: python3 -m http.server
```

Open `demo/index.html`, or open the self-contained
`demo/example_output.html` directly in a browser (CDN access still required).

## Jupyter

```python
from open_fret.viewer import build_view_bundle, show
bundle = build_view_bundle(midi_tokens=[...], tab_tokens=[...])
show(bundle)   # needs npm run build to have been run once
```

## Status / limits

- Browser playback QA (click everything, listen) still needs a human pass —
  see Issue #13. Static checks: component builds, bundle JSON validates,
  example page contains all mount points and scripts.
- Whole-GPU/per-tab audio depends on CDN + soundfont reachability; offline
  use falls back to token inspectors with a warning banner.
