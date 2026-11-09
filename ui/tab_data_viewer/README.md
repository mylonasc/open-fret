# tab_data_viewer

Playable + seekable React viewer for open-fret model I/O pairs (Issue #13).

- **Input:** MIDI notes (audio via `html-midi-player`, visualization via
  **WaveRoll**'s `createPianoRoll`).
- **Output:** predicted TAB tokens rendered by **AlphaTab** (notation + tab,
  audio playback, beat cursor, click-the-score to seek).
- Embeddable in a web page (`demo/index.html`) or a Jupyter cell via
  `open_fret.viewer.show()`.

## Confirmed CDN endpoints (checked 2026-11-09)

- AlphaTab: `https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/alphaTab.min.js`
- SoundFont: `.../dist/soundfont/sonivox.sf2`
- WaveRoll: `https://cdn.jsdelivr.net/npm/wave-roll@0.4.0/dist/wave-roll.umd.js`
- MIDI stack: jsDelivr `combine/` of `tone@14.7.77` +
  `@magenta/music@1.23.0/es6/core.js` + `focus-visible@5` +
  `html-midi-player@1.5.0` (note: `@magenta/music@1.23.2` does **not** exist;
  `html-midi-player@1.6.0` changes the dist layout, so 1.5.0 is pinned).

## Verified API facts (from upstream docs, not guesses)

- `html-midi-player`: `<midi-player src visualizer="#id" sound-font>`,
  JS `.start()` / `.stop()`, `.currentTime`, `.playing`,
  `load`/`note`/`start`/`stop` events. The seek bar lives in its open
  shadow root; the viewer mirrors `input`/`change` to the piano roll.
- WaveRoll: `window.WaveRoll.createPianoRoll(container, notes, options)`
  resolves to an instance with `setTime(seconds)`, `resize(width, height)`,
  `destroy()`. Notes are plain objects with `midi`, `time` (seconds),
  `duration` (seconds), `ticks`, `name`, `pitch`, `octave`, `velocity`.
  `setTime` keeps the playhead at the left edge and scrolls the roll.
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

## Status

Browser-verified with headless Chrome (Playwright, `e2e_check.py` in this
folder, 13 assertions): notation + tab staves render, MIDI plays with
advancing position, tab plays with advancing tick readout, track mute
toggle works, play-both starts the MIDI side, click-to-seek moves the
playhead, zoom (A−/A+/reset) re-layouts the score, token inspectors
render — zero JS errors, zero failed requests. Real bugs this caught and
fixed:
1. Vite lib build leaking `process.env.NODE_ENV` (fixed with explicit
   `define`; bundle dropped 478KB → 149KB as a bonus),
2. mount script running before the dist bundle (script order),
3. `settings.tex` is a *boolean* — tex content loads via `api.tex()`,
4. our AlphaTex header was rejected by AlphaTab's importer: blocks are
   line-oriented and `\track` needs a quoted name (header now mirrors the
   upstream guitartab format).
- Audio depends on CDN + soundfont reachability; offline use falls back to
  token inspectors with a warning banner.
