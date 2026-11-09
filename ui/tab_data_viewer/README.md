# tab_data_viewer

Playable + seekable React viewer for open-fret model I/O pairs (Issue #13).

- **Single transport:** AlphaTab owns the score clock and audio. One
  Play/Pause button, one seek bar, score click-to-seek and piano-roll
  click-to-seek all move the tab cursor and the roll playhead together.
- **Listen toggle:** flip between the **Tab** rendering and the **MIDI**
  rendering of the exact input notes (`midi_alpha_tex` in the bundle). Both
  play through the same AlphaTab player, so the A/B comparison cannot drift;
  position is preserved and playback resumes across the flip.
- **Score cursor:** a green bar (`at-cursor-bar`) tracks the sounding
  measures plus a beat line and note highlighting — site CSS AlphaTab needs
  but does not ship.
- **Input view:** the source MIDI notes drawn by **WaveRoll**'s
  `createPianoRoll` (grid, C-octave labels, zoom/pan, scrolling playhead).
- **Output view:** predicted TAB tokens rendered by **AlphaTab** (notation +
  tab, audio playback, beat cursor).
- Embeddable in a web page (`demo/index.html`) or a Jupyter cell via
  `open_fret.viewer.show()`.

## Confirmed CDN endpoints (checked 2026-11-09)

- AlphaTab: `https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/alphaTab.min.js`
- SoundFont: `.../dist/soundfont/sonivox.sf2`
- WaveRoll: `https://cdn.jsdelivr.net/npm/wave-roll@0.4.0/dist/wave-roll.umd.js`

## Verified API facts (from upstream docs, not guesses)

- AlphaTab UMD exposes `window.alphaTab.AlphaTabApi`; the single transport
  uses `playPause()`/`stop()`, `tickPosition` (score ticks, 960 per quarter),
  `playerStateChanged` and `playerPositionChanged` (milliseconds + ticks).
  The tempo selector maps to `playbackSpeed`; the tick grid is
  tempo-independent so the roll stays synced at any tempo. Score clicks seek
  via `beatMouseDown` → `playbackStart`. The Listen toggle swaps the loaded
  score between the tab AlphaTex and the bundle's pitched MIDI rendering
  (`midi_alpha_tex`, a `\staff {score}` track with `{t}` ties) via `tex()`,
  restoring `tickPosition` afterwards — one player, one clock, no MIDI
  importer needed (the 1.3.0 JS build ships no MIDI importer).
  Everything else (mute, tick position, beat events) is **feature-detected**
  in the component — missing APIs degrade to warnings, not crashes.
- WaveRoll: `window.WaveRoll.createPianoRoll(container, notes, options)`
  resolves to an instance with `setTime(seconds)`, `resize(width, height)`,
  `destroy()`. Notes are plain objects with `midi`, `time` (seconds),
  `duration` (seconds), `ticks`, `name`, `pitch`, `octave`, `velocity`.
  `setTime` keeps the playhead at the left edge and scrolls the roll. Roll
  clicks are converted back to seconds by inverting the internal time scale
  (piano-key gutter + pan/zoom aware) and seek the same transport.

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
