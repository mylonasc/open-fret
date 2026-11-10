import React from "react";
import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const SOUNDFONT_URL =
  "https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/soundfont/sonivox.sf2";

function embeddedSoundFontUrl() {
  // Notebook/offline embedding (open_fret.viewer inline_deps): the page
  // creates a blob URL from the vendored soundfont at load time.
  try {
    const url = window.__OFV_SOUNDFONT_URL;
    if (typeof url === "string" && url.startsWith("blob:")) return url;
  } catch (_) {}
  return null;
}

function embeddedScriptFile() {
  // AlphaTab's worker bootstrap needs a real script URL; the inline loader
  // (inline_deps pages) stashes a data: URL here. CDN pages use detection.
  try {
    const url = window.__OFV_AT_SCRIPT_URL;
    if (typeof url === "string" && url.length > 100) return url;
  } catch (_) {}
  return null;
}
const PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function tokensToNotes(tokens) {
  // Fallback for bundles created before input_notes was added.
  let time = 0;
  const open = new Map();
  const notes = [];
  for (const token of tokens || []) {
    const value = String(token);
    if (value.startsWith("TS_")) {
      const delta = Number.parseInt(value.slice(3), 10);
      if (Number.isFinite(delta)) time += delta;
    } else if (value.startsWith("NO_")) {
      const pitch = Number.parseInt(value.slice(3), 10);
      if (!Number.isFinite(pitch)) continue;
      if (!open.has(pitch)) open.set(pitch, []);
      open.get(pitch).push(time);
    } else if (value.startsWith("NF_")) {
      const pitch = Number.parseInt(value.slice(3), 10);
      const stack = open.get(pitch);
      if (stack && stack.length) {
        notes.push({ start: stack.pop(), end: time, pitch });
      }
    }
  }
  for (const [pitch, stack] of open.entries()) {
    for (const start of stack) {
      notes.push({ start, end: Math.max(time, start + 480), pitch });
    }
  }
  return notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

function rollNotes(bundle) {
  const ticksPerBeat = bundle.ticks_per_beat || 480;
  const tempo = bundle.tempo || 120;
  const secondsPerTick = 60 / (tempo * ticksPerBeat);
  const source =
    Array.isArray(bundle.input_notes) && bundle.input_notes.length
      ? bundle.input_notes
      : tokensToNotes(bundle.input_tokens);
  return source.map((note) => {
    const midi = note.pitch;
    const start = note.start;
    const end = Number.isFinite(note.end) ? note.end : start + 480;
    return {
      midi,
      time: start * secondsPerTick,
      ticks: start,
      name: `${PITCH_NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`,
      pitch: PITCH_NAMES[midi % 12],
      octave: Math.floor(midi / 12) - 1,
      velocity: 0.8,
      duration: Math.max(1, end - start) * secondsPerTick,
    };
  });
}

const PIANO_ROLL_OPTIONS = {
  showPianoKeys: true,
  showWaveformBand: false,
  backgroundColor: 0xffffff,
  noteColor: 0x4f46e5,
  playheadColor: 0x0f172a,
  timeStep: 1,
  minorTimeStep: 0.25,
};

// AlphaTab counts 960 ticks per quarter note internally (verified live:
// endTick/endTime == tempo*960/60 ticks per second). Source token windows use
// bundle.ticks_per_beat (480) — the two grids agree on musical time because
// both are rendered at the same source tempo.
const SCORE_TICKS_PER_QUARTER = 960;

function scoreTickToSeconds(tick, bundle) {
  return ((Number(tick) || 0) * 60) / ((bundle.tempo || 120) * SCORE_TICKS_PER_QUARTER);
}

function scoreSecondsToTick(seconds, bundle) {
  return Math.max(
    0,
    Math.round(((Number(seconds) || 0) * (bundle.tempo || 120) * SCORE_TICKS_PER_QUARTER) / 60)
  );
}

function sourceTickToSeconds(tick, bundle) {
  return ((Number(tick) || 0) * 60) / ((bundle.tempo || 120) * (bundle.ticks_per_beat || 480));
}

function sourceNotes(bundle) {
  return Array.isArray(bundle.input_notes) && bundle.input_notes.length
    ? bundle.input_notes
    : tokensToNotes(bundle.input_tokens);
}

function fallbackDuration(bundle) {
  const ends = sourceNotes(bundle).map((n) => Number(n.end) || 0);
  return ends.length ? sourceTickToSeconds(Math.max(...ends), bundle) : 0;
}

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const rest = value - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

function rollTimeAtClientX(roll, host, clientX) {
  // WaveRoll paints time as scale(t) * zoomX + pianoKeys + panX. Invert that
  // expression so a click lands on the note grid under the pointer.
  const inner = roll && roll._instance;
  if (!inner || !inner.timeScale || !inner.state) return null;
  try {
    const rect = host.getBoundingClientRect();
    const keys = inner.options.showPianoKeys ? 60 : 0;
    const x = clientX - rect.left;
    const time = inner.timeScale.invert(
      (x - keys - inner.state.panX) / inner.state.zoomX
    );
    const end = inner.timeScale.domain()[1];
    return Math.min(Math.max(time, 0), end);
  } catch (_) {
    return null;
  }
}

function pianoRollOptions(notes) {
  const pitches = notes.map((n) => n.midi).filter(Number.isFinite);
  let min = pitches.length ? Math.min(...pitches) : 40;
  let max = pitches.length ? Math.max(...pitches) : 88;
  // Keep at least an octave of context so a single-note window is readable.
  if (max - min < 12) {
    const pad = Math.ceil((12 - (max - min)) / 2);
    min -= pad;
    max += pad;
  }
  return {
    ...PIANO_ROLL_OPTIONS,
    noteRange: {
      min: Math.max(0, min - 1),
      max: Math.min(127, max + 1),
    },
  };
}

const CSS = `
.ofv { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #1c1c1e; max-width: 1400px; }
.ofv-header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin: 2px 0 10px; }
.ofv-title { font-size: 18px; margin: 0; font-weight: 650; }
.ofv-badges { display: flex; gap: 6px; flex-wrap: wrap; }
.ofv-badge { font-size: 12px; background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe;
  border-radius: 999px; padding: 2px 10px; white-space: nowrap; }
.ofv-warn { background: #fff4e5; border: 1px solid #f5c97b; border-radius: 8px; padding: 8px 12px; margin-bottom: 10px; font-size: 13px; }
.ofv-transport { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px 12px; margin-bottom: 12px; }
.ofv-btn { border: 1px solid #cbd5e1; background: #fff; border-radius: 8px; padding: 6px 12px;
  font-size: 13px; cursor: pointer; }
.ofv-btn:disabled { opacity: 0.45; cursor: default; }
.ofv-btn.primary { background: #4f46e5; border-color: #4f46e5; color: #fff; font-weight: 600; }
.ofv-btn:not(:disabled):hover { border-color: #4f46e5; }
.ofv-tempo { font-size: 13px; color: #475569; display: flex; align-items: center; gap: 6px; }
.ofv-tempo select { border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 6px; font-size: 13px; }
.ofv-tick { font-size: 12px; color: #64748b; font-variant-numeric: tabular-nums; }
.ofv-grid { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); gap: 12px; }
@media (max-width: 760px) { .ofv-grid { grid-template-columns: 1fr; } }
.ofv-card { border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; background: #fff;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); min-width: 0; }
.ofv-card h4 { margin: 0; padding: 8px 12px; font-size: 13px; font-weight: 650;
  background: #f8fafc; border-bottom: 1px solid #e2e8f0; }
.ofv-midi-body { padding: 10px 12px; min-height: 120px; }
.ofv-piano-roll { width: 100%; height: 300px; border: 1px solid #e2e8f0;
  border-radius: 8px; overflow: hidden; background: #fff; position: relative; }
@media (max-width: 760px) { .ofv-piano-roll { height: 220px; } }
.ofv-seek-row { display: flex; align-items: center; gap: 10px; width: 100%;
  padding-top: 8px; border-top: 1px solid #e2e8f0; margin-top: 2px; }
.ofv-seek-label { font-size: 12px; color: #64748b; text-transform: uppercase;
  letter-spacing: 0.06em; }
.ofv-seek { flex: 1 1 240px; min-width: 120px; accent-color: #4f46e5; cursor: pointer; }
.ofv-seek:disabled { cursor: default; opacity: 0.5; }
.ofv-time { font-size: 12px; color: #475569; font-variant-numeric: tabular-nums;
  white-space: nowrap; }
.ofv-midi-body .ofv-skeleton { position: absolute; inset: 0; display: flex;
  align-items: center; justify-content: center; }
.ofv-roll-wrap { position: relative; }
.ofv-card-head { display: flex; align-items: center; justify-content: space-between;
  gap: 8px; padding: 6px 12px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }
.ofv-card-head h4 { margin: 0; padding: 0; background: none; border: none;
  font-size: 13px; font-weight: 650; }
.ofv-tab-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  justify-content: flex-end; }
.ofv-mode { display: inline-flex; border: 1px solid #cbd5e1; border-radius: 8px;
  overflow: hidden; flex-shrink: 0; }
.ofv-mode button { border: 0; background: #fff; color: #334155; padding: 5px 10px;
  font-size: 12px; cursor: pointer; }
.ofv-mode button + button { border-left: 1px solid #cbd5e1; }
.ofv-mode button.active { background: #4f46e5; color: #fff; font-weight: 650; }
.ofv-mode button:disabled { opacity: 0.45; cursor: default; }
.ofv-zoom { display: flex; gap: 4px; align-items: center; flex-shrink: 0; }
.ofv-zoom button { border: 1px solid #cbd5e1; background: #fff; border-radius: 6px;
  padding: 2px 9px; font-size: 12px; cursor: pointer; line-height: 1.6; }
.ofv-zoom button:disabled { opacity: 0.45; cursor: default; }
.ofv-zoom-label { font-size: 12px; color: #475569; min-width: 46px; text-align: center;
  font-variant-numeric: tabular-nums; cursor: pointer; border: none; background: none; }
.ofv-tab-scroll { overflow: auto; max-height: 70vh; min-height: 200px; max-width: 100%;
  resize: vertical; }
.ofv-tab-render { width: 100%; min-width: 0; }
/* AlphaTab playback cursor: a green bar over the sounding measures plus a
   thin beat line. AlphaTab adds these divs itself (enableCursor); without
   site CSS they are invisible, which is why the tab showed no progress. */
.ofv-tab-render .at-cursor-bar { background: rgba(34, 197, 94, 0.30); }
.ofv-tab-render .at-cursor-beat { background: rgba(21, 128, 61, 0.95); width: 3px; }
.ofv-tab-render .at-highlight * { fill: #16a34a; stroke: #16a34a; }
.ofv-skeleton { padding: 24px; color: #94a3b8; font-size: 13px; text-align: center;
  background: repeating-linear-gradient(45deg, #f8fafc, #f8fafc 10px, #f1f5f9 10px, #f1f5f9 20px); }
.ofv-tokens { margin-top: 10px; }
.ofv-tokens summary { cursor: pointer; font-size: 13px; color: #475569; }
.ofv-tokens pre { white-space: pre-wrap; word-break: break-word; background: #f8fafc;
  border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px; max-height: 160px;
  overflow: auto; font-size: 12px; }
.ofv-foot { margin-top: 8px; font-size: 12px; color: #94a3b8; }
`;

function TokenInspector({ title, tokens }) {
  const text = Array.isArray(tokens) ? tokens.join(" ") : String(tokens ?? "");
  return (
    <details className="ofv-tokens">
      <summary>
        {title} ({Array.isArray(tokens) ? tokens.length : 0} tokens)
      </summary>
      <pre>{text}</pre>
    </details>
  );
}

export function TabDataViewer({ bundle }) {
  const rollWrapRef = useRef(null);
  const tabScrollRef = useRef(null);
  const tabRenderRef = useRef(null);
  const apiRef = useRef(null);
  const rollRef = useRef(null);
  const trackRef = useRef([]);
  const [ready, setReady] = useState({ tab: false, player: false, roll: false });
  const [tabPlaying, setTabPlaying] = useState(false);
  const [tabMode, setTabMode] = useState("predicted");
  const [audioMode, setAudioMode] = useState("tab");
  const pendingTickRef = useRef(null);
  const resumeRef = useRef(false);
  const hasReference = Boolean(bundle.reference_alpha_tex);
  const activeTabMode = tabMode === "actual" && hasReference ? "actual" : "predicted";
  const activeTex = activeTabMode === "actual" ? bundle.reference_alpha_tex : bundle.alpha_tex;
  const [muted, setMuted] = useState(false);
  const [tempo, setTempo] = useState(bundle.tempo || 120);
  const [pos, setPos] = useState(0);
  const [duration, setDuration] = useState(() => fallbackDuration(bundle));
  const [zoom, setZoom] = useState(1);
  const zoomRef = useRef(1);
  const applyZoom = (z) => {
    const clamped = Math.min(2, Math.max(0.5, Math.round(z * 20) / 20));
    zoomRef.current = clamped;
    setZoom(clamped);
    const api = apiRef.current;
    if (!api) return;
    try {
      if (api.settings && api.settings.display) {
        api.settings.display.scale = clamped;
        // ScoreRenderer.updateSettings only stores settings — render() is
        // what re-layouts at the new scale.
        if (typeof api.updateSettings === "function") api.updateSettings();
        if (typeof api.render === "function") api.render();
      } else {
        addWarn("This AlphaTab build exposes no display scale setting.");
      }
    } catch (err) {
      addWarn(`Zoom failed: ${err}`);
    }
  };
  const [warn, setWarn] = useState([]);

  const addWarn = (m) => setWarn((w) => (w.includes(m) ? w : [...w, m]));
  const uid = bundle._uid || "x";

  const syncRoll = (seconds) => {
    const roll = rollRef.current;
    if (roll && Number.isFinite(seconds)) {
      try {
        roll.setTime(seconds);
      } catch (_) {}
    }
  };

  // Single transport: AlphaTab owns the score clock and audio. Every seek
  // (slider, score click, piano-roll click) goes through here so the tab and
  // the roll cannot drift apart.
  const seekTo = (seconds) => {
    const value = Math.max(0, Number(seconds) || 0);
    const api = apiRef.current;
    pendingTickRef.current = null;
    try {
      if (api) api.tickPosition = scoreSecondsToTick(value, bundle);
    } catch (_) {}
    setPos(value);
    syncRoll(value);
  };

  const refreshDuration = () => {
    const api = apiRef.current;
    let seconds = 0;
    try {
      const bars = api && api.tickCache && api.tickCache.masterBars;
      if (bars && bars.length) {
        seconds = scoreTickToSeconds(bars[bars.length - 1].end, bundle);
      }
    } catch (_) {}
    if (!(seconds > 0)) seconds = fallbackDuration(bundle);
    if (seconds > 0) setDuration(seconds);
    return seconds;
  };

  // Piano roll: WaveRoll draws the grid, notes and a scrolling playhead. Its
  // position is driven by the AlphaTab tick position (see the rAF effect).
  useEffect(() => {
    const host = rollWrapRef.current;
    if (!host) return;
    let roll = null;
    let resizeObserver = null;
    let cancelled = false;
    let down = null;
    const onPointerDown = (event) => {
      down = { x: event.clientX, y: event.clientY };
    };
    const onClick = (event) => {
      if (event.target && event.target.closest && event.target.closest("button")) return;
      if (down && Math.hypot(event.clientX - down.x, event.clientY - down.y) > 6) return;
      const time = rollTimeAtClientX(roll, host, event.clientX);
      if (time != null) seekTo(time);
    };
    host.addEventListener("pointerdown", onPointerDown);
    host.addEventListener("click", onClick);
    try {
      if (!window.WaveRoll || !window.WaveRoll.createPianoRoll) {
        addWarn("WaveRoll failed to load (CDN blocked?). Piano roll disabled.");
      } else {
        const notes = rollNotes(bundle);
        window.WaveRoll.createPianoRoll(
          host,
          notes,
          pianoRollOptions(notes)
        ).then((instance) => {
          if (cancelled) {
            try { instance.destroy(); } catch (_) {}
            return;
          }
          roll = instance;
          rollRef.current = instance;
          try { host._ofvRoll = instance; } catch (_) {}
          setReady((r) => ({ ...r, roll: true }));
          let start = 0;
          try {
            if (apiRef.current) {
              start = scoreTickToSeconds(apiRef.current.tickPosition, bundle);
            }
          } catch (_) {}
          try { instance.setTime(start); } catch (_) {}
          resizeObserver = new ResizeObserver(() => {
            try {
              instance.resize(host.clientWidth, host.clientHeight);
            } catch (_) {}
          });
          resizeObserver.observe(host);
        }).catch((err) => {
          addWarn(`Piano roll failed: ${err}`);
          console.error(err);
        });
      }
    } catch (err) {
      addWarn(`Piano roll failed: ${err}`);
      console.error(err);
    }
    return () => {
      cancelled = true;
      host.removeEventListener("pointerdown", onPointerDown);
      host.removeEventListener("click", onClick);
      if (resizeObserver) resizeObserver.disconnect();
      try {
        // WaveRoll 0.4.0's destroy() does not cancel its throttled render
        // frame; a queued frame firing post-destroy throws on nulled fields.
        // Cancel it first (internal handle, guarded: harmless if renamed).
        const inst = roll && roll._instance;
        if (inst && inst.rafId != null && typeof cancelAnimationFrame === "function") {
          cancelAnimationFrame(inst.rafId);
          inst.rafId = null;
        }
      } catch (_) {}
      try {
        roll && roll.destroy && roll.destroy();
      } catch (_) {}
      rollRef.current = null;
      host.innerHTML = "";
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bundle.midi_b64, bundle.title, bundle.tempo]);

  // TAB panel: render into the INNER node; the OUTER node is the scroll
  // element. (Passing the render node as its own scrollElement makes the
  // score grow wider on every render cycle.)
  //
  // The AlphaTabApi (and its player/soundfont) is created once per bundle and
  // reused when swapping predicted/actual: only ``tex()`` changes the score.
  // Recreating the player on each switch loses the seek position because the
  // fresh player is not ready yet when we try to restore it.
  useEffect(() => {
    const el = tabRenderRef.current;
    if (!el) return;
    el.innerHTML = "";
    const AT = window.alphaTab;
    if (!AT || !AT.AlphaTabApi) {
      addWarn("AlphaTab failed to load (CDN blocked?). Tab panel disabled.");
      return;
    }
    let api = null;
    try {
      const scriptFile = embeddedScriptFile();
      const options = {
        player: {
          soundFont: embeddedSoundFontUrl() || SOUNDFONT_URL,
          enablePlayer: true,
          enableCursor: true,
          scrollElement: tabScrollRef.current || el,
        },
      };
      // Offline/blob-loaded AlphaTab cannot detect its own script file for
      // the render worker; the inline page supplies a data: URL instead.
      if (scriptFile) {
        options.core = { scriptFile };
        // Reliability over latency offline: ScriptProcessor output needs no
        // worklet module loading (WebAudioScriptProcessor = 1); the CDN
        // default keeps audio worklets.
        options.player.outputMode = 1;
      }
      api = new AT.AlphaTabApi(el, options);
      apiRef.current = api;
      try {
        el._ofvApi = api;
      } catch (_) {} // test/debug hook (e2e reads tickPosition)
      try {
        api.error &&
          api.error.on &&
          api.error.on((e) =>
            addWarn(`AlphaTab score error: ${e && e.message ? e.message : e}`)
          );
      } catch (_) {}
      try {
        api.scoreLoaded &&
          api.scoreLoaded.on &&
          api.scoreLoaded.on((score) => {
            setReady((r) => ({ ...r, tab: true }));
            try {
              trackRef.current = score && score.tracks ? score.tracks : [];
            } catch (_) {}
            refreshDuration();
          });
      } catch (_) {}
      try {
        api.playerReady &&
          api.playerReady.on &&
          api.playerReady.on(() => {
            setReady((r) => ({ ...r, player: true }));
            refreshDuration();
          });
      } catch (_) {}
      // AlphaTab is the only audio source and the only clock: its state and
      // tick position drive the transport bar, the tab cursor and the roll.
      try {
        api.playerStateChanged &&
          api.playerStateChanged.on &&
          api.playerStateChanged.on((e) => {
            setTabPlaying(Boolean(e && e.state === 1));
            if (e && e.stopped) {
              setPos(0);
              syncRoll(0);
            }
          });
      } catch (_) {}
      try {
        api.playerPositionChanged &&
          api.playerPositionChanged.on &&
          api.playerPositionChanged.on((args) => {
            if (!args || pendingTickRef.current != null) return;
            const seconds = scoreTickToSeconds(args.currentTick, bundle);
            setPos(seconds);
            syncRoll(seconds);
            if (args.endTick > 0) {
              setDuration(scoreTickToSeconds(args.endTick, bundle));
            }
          });
      } catch (_) {}
      // Re-apply the user's zoom on (re)mount — a fresh Api starts at 1.0.
      try {
        if (api.settings && api.settings.display) {
          api.settings.display.scale = zoomRef.current;
        }
      } catch (_) {}
      // Click-to-seek: AlphaTab doesn't move the playhead on click by
      // itself — beatMouseDown gives the beat, playbackStart is its tick.
      try {
        if (api.beatMouseDown && api.beatMouseDown.on) {
          api.beatMouseDown.on((beat) => {
            try {
              if (beat && typeof beat.playbackStart === "number") {
                seekTo(scoreTickToSeconds(beat.playbackStart, bundle));
              }
            } catch (_) {}
          });
        }
      } catch (_) {}
    } catch (err) {
      addWarn(`AlphaTab failed to render this score: ${err}`);
      console.error(err);
    }
    return () => {
      try {
        api && api.destroy && api.destroy();
      } catch (_) {}
      apiRef.current = null;
      trackRef.current = [];
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bundle.title]);

  // Roll + slider follow the score clock smoothly while playing. Position
  // events alone are too coarse for a scrolling playhead, so mirror the
  // AlphaTab tick each animation frame (score seconds; unaffected by tempo).
  useEffect(() => {
    if (!tabPlaying) return;
    let raf = null;
    let lastUi = 0;
    const loop = (now) => {
      const api = apiRef.current;
      if (api && pendingTickRef.current == null) {
        let tick = null;
        try {
          tick = api.tickPosition;
        } catch (_) {}
        if (Number.isFinite(tick)) {
          const seconds = scoreTickToSeconds(tick, bundle);
          syncRoll(seconds);
          // The roll moves every frame; React state only needs ~10 Hz.
          if (now - lastUi >= 100) {
            lastUi = now;
            setPos(seconds);
          }
        }
      }
      raf = window.requestAnimationFrame(loop);
    };
    raf = window.requestAnimationFrame(loop);
    return () => {
      if (raf != null) window.cancelAnimationFrame(raf);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tabPlaying]);

  // Swap the score when the user toggles predicted/actual or flips the
  // Listen source between tab and MIDI, keeping the same player so
  // playback/seeking survives the switch. The MIDI score is a pitched
  // rendering of the exact input notes (no second player/clock).
  const scoreKey = audioMode === "midi" ? `midi:${bundle.title}` : activeTex;
  useEffect(() => {
    const api = apiRef.current;
    if (!api) return;
    try {
      if (api.settings && api.settings.display) {
        api.settings.display.scale = zoomRef.current;
      }
      if (audioMode === "midi") {
        if (!bundle.midi_alpha_tex) {
          throw new Error("This bundle has no input-MIDI rendering (rebuild it).");
        }
        api.tex(bundle.midi_alpha_tex);
      } else {
        api.tex(activeTex);
      }
    } catch (err) {
      addWarn(`AlphaTab failed to render this score: ${err}`);
      if (audioMode === "midi") setAudioMode("tab");
      return;
    }

    const target = pendingTickRef.current;
    if (target == null) {
      resumeRef.current = false;
      refreshDuration();
      return;
    }

    let pollTimer = null;
    let settleTimer = null;
    let giveUpTimer = null;
    const stop = () => {
      if (pollTimer) clearInterval(pollTimer);
      if (settleTimer) clearTimeout(settleTimer);
      if (giveUpTimer) clearTimeout(giveUpTimer);
      pollTimer = settleTimer = giveUpTimer = null;
    };
    const verify = () => {
      if (pendingTickRef.current == null) {
        stop();
        return;
      }
      try {
        api.tickPosition = target;
        if (Math.abs((api.tickPosition || 0) - target) > 2) return;
      } catch (_) {
        return;
      }
      if (settleTimer) return;
      // A freshly loaded score may accept a position and then reset it while
      // the player initialises, so require it to stay put briefly.
      settleTimer = setTimeout(() => {
        settleTimer = null;
        try {
          if (
            pendingTickRef.current != null &&
            Math.abs((api.tickPosition || 0) - target) <= 2
          ) {
            setPos(scoreTickToSeconds(target, bundle));
            syncRoll(scoreTickToSeconds(target, bundle));
            pendingTickRef.current = null;
            stop();
            finishRestore();
          }
        } catch (_) {}
      }, 400);
    };
    const listen = (name, fn) => {
      try {
        api[name] && api[name].on && api[name].on(fn);
      } catch (_) {}
    };
    const unlisten = (name, fn) => {
      try {
        api[name] && api[name].off && api[name].off(fn);
      } catch (_) {}
    };
    for (const name of ["scoreLoaded", "renderFinished", "playerReady"]) {
      listen(name, verify);
    }
    pollTimer = setInterval(verify, 250);
    const finishRestore = () => {
      // One-click A/B: if playback was running before the switch, resume it
      // once the new score holds its position.
      if (resumeRef.current) {
        resumeRef.current = false;
        try {
          const api = apiRef.current;
          if (api && typeof api.play === "function") {
            api.play();
            setTabPlaying(true);
          }
        } catch (_) {}
      }
    };
    giveUpTimer = setTimeout(() => {
      pendingTickRef.current = null;
      resumeRef.current = false;
      stop();
    }, 15000);
    return () => {
      stop();
      for (const name of ["scoreLoaded", "renderFinished", "playerReady"]) {
        unlisten(name, verify);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scoreKey, bundle.title]);

  const togglePlay = () => {
    const api = apiRef.current;
    if (!api) return;
    try {
      if (typeof api.playPause === "function") api.playPause();
      else if (tabPlaying && typeof api.pause === "function") api.pause();
      else if (!tabPlaying && typeof api.play === "function") api.play();
      else {
        addWarn("This AlphaTab build exposes no play method.");
        return;
      }
      // playerStateChanged corrects this if the player disagrees.
      setTabPlaying(!tabPlaying);
    } catch (err) {
      addWarn(`Tab playback failed: ${err}`);
    }
  };
  const stopAll = () => {
    const api = apiRef.current;
    try {
      api && api.stop && api.stop();
    } catch (_) {}
    setTabPlaying(false);
    setPos(0);
    syncRoll(0);
  };
  const onSeek = (event) => {
    seekTo(Number(event.target.value));
  };
  const switchTabMode = (mode) => {
    if (audioMode === "midi") {
      // Choosing a fingering implies listening to the tab again.
      switchAudioMode("tab");
    }
    if (mode === activeTabMode || (mode === "actual" && !hasReference)) return;
    try {
      pendingTickRef.current =
        apiRef.current && typeof apiRef.current.tickPosition === "number"
          ? apiRef.current.tickPosition
          : scoreSecondsToTick(pos, bundle);
    } catch (_) {
      pendingTickRef.current = scoreSecondsToTick(pos, bundle);
    }
    resumeRef.current = tabPlaying;
    try {
      const api = apiRef.current;
      if (api) {
        if (typeof api.pause === "function") api.pause();
        else if (api.stop) api.stop();
      }
    } catch (_) {}
    setTabPlaying(false);
    setReady((r) => ({ ...r, tab: false }));
    setTabMode(mode);
  };
  const hasMidiTex = Boolean(bundle.midi_alpha_tex);
  const switchAudioMode = (mode) => {
    if (mode === audioMode) return;
    if (mode === "midi" && !hasMidiTex) {
      addWarn("This bundle has no input-MIDI rendering (rebuild it).");
      return;
    }
    try {
      pendingTickRef.current =
        apiRef.current && typeof apiRef.current.tickPosition === "number"
          ? apiRef.current.tickPosition
          : scoreSecondsToTick(pos, bundle);
    } catch (_) {
      pendingTickRef.current = scoreSecondsToTick(pos, bundle);
    }
    resumeRef.current = tabPlaying;
    try {
      const api = apiRef.current;
      if (api) {
        if (typeof api.pause === "function") api.pause();
        else if (api.stop) api.stop();
      }
    } catch (_) {}
    setTabPlaying(false);
    setReady((r) => ({ ...r, tab: false }));
    setAudioMode(mode);
  };
  const toggleMute = () => {
    const api = apiRef.current;
    const tracks = trackRef.current;
    if (!api || !tracks.length) {
      addWarn("Mute needs a loaded score (wait for the tab to render).");
      return;
    }
    try {
      if (typeof api.changeTrackMute === "function") {
        api.changeTrackMute(tracks, !muted);
        setMuted(!muted);
      } else {
        addWarn("This AlphaTab build exposes no changeTrackMute.");
      }
    } catch (err) {
      addWarn(`Mute failed: ${err}`);
    }
  };
  const onTempo = (v) => {
    const t = Number(v);
    setTempo(t);
    try {
      const api = apiRef.current;
      if (api && "playbackSpeed" in api) api.playbackSpeed = t / (bundle.tempo || 120);
      else addWarn("Tempo scaling needs AlphaTab playbackSpeed (not in this build).");
    } catch (_) {}
  };

  const meta = bundle.meta || {};
  return (
    <div className="ofv">
      <style>{CSS}</style>
      <div className="ofv-header">
        <h3 className="ofv-title">{bundle.title || "open-fret sample"}</h3>
        <div className="ofv-badges">
          {meta.n_input_notes != null && <span className="ofv-badge">🎹 {meta.n_input_notes} notes in</span>}
          {meta.n_tab_tokens != null && <span className="ofv-badge">🎸 {meta.n_tab_tokens} predicted tab tokens</span>}
          {hasReference && meta.n_reference_tab_tokens != null && (
            <span className="ofv-badge">🎼 {meta.n_reference_tab_tokens} actual tab tokens</span>
          )}
          {bundle.tempo && <span className="ofv-badge">♪ {bundle.tempo} BPM source</span>}
        </div>
      </div>
      {warn.length > 0 && (
        <div className="ofv-warn">
          {warn.map((w, i) => (
            <div key={i}>⚠️ {w}</div>
          ))}
        </div>
      )}
      <div className="ofv-transport">
        <button
          className="ofv-btn primary"
          onClick={togglePlay}
          disabled={!ready.tab || !ready.player}
          title={ready.player ? "Play or pause (tab audio + roll follow)" : "Waiting for the tab player…"}
        >
          {tabPlaying ? "⏸ Pause" : "▶ Play"}
        </button>
        <button
          className="ofv-btn"
          onClick={stopAll}
          disabled={!ready.tab}
          title="Stop and return to the start"
        >
          ⏹
        </button>
        <span className="ofv-seek-label">seek</span>
        <input
          type="range"
          className="ofv-seek"
          aria-label="Seek tablature and piano roll"
          min={0}
          max={duration > 0 ? duration : 0}
          step={0.01}
          value={duration > 0 ? Math.min(pos, duration) : 0}
          onChange={onSeek}
          disabled={!ready.tab || !(duration > 0)}
        />
        <span className="ofv-time">
          {formatTime(pos)} / {formatTime(duration)}
        </span>
        <button
          className="ofv-btn"
          onClick={toggleMute}
          disabled={!ready.tab}
          title="Mute/unmute the sounding score for A/B listening"
        >
          {muted ? "🔈 Unmute" : "🔇 Mute"}
        </button>
        <label className="ofv-tempo">
          tempo
          <select value={tempo} onChange={(e) => onTempo(e.target.value)}>
            {[60, 80, 100, 120, 140, 160].map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <span className="ofv-tick">tick {scoreSecondsToTick(pos, bundle)}</span>
      </div>
      <div className="ofv-grid">
        <div className="ofv-card">
          <h4>Input MIDI — piano roll (click to seek; drag to pan, wheel to zoom)</h4>
          <div className="ofv-midi-body">
            <div className="ofv-roll-wrap">
              {!ready.roll && <div className="ofv-skeleton">Loading piano roll…</div>}
              <div ref={rollWrapRef} className="ofv-piano-roll" data-uid={uid}
                role="img" aria-label="Input MIDI piano roll" />
            </div>
          </div>
        </div>
        <div className="ofv-card">
          <div className="ofv-card-head">
            <h4>{audioMode === "midi"
              ? "Input MIDI — score (same notes as the piano roll)"
              : `${activeTabMode === "actual" ? "Actual reference tab" : "Predicted tab"} — click the score to seek`}</h4>
            <div className="ofv-tab-controls">
              <div className="ofv-mode" role="group" aria-label="Choose what to listen to">
                <button
                  type="button"
                  className={audioMode === "tab" ? "active" : ""}
                  aria-pressed={audioMode === "tab"}
                  title="Listen to the tablature audio"
                  onClick={() => switchAudioMode("tab")}
                >
                  Tab
                </button>
                <button
                  type="button"
                  className={audioMode === "midi" ? "active" : ""}
                  aria-pressed={audioMode === "midi"}
                  disabled={!hasMidiTex}
                  title={hasMidiTex
                    ? "Listen to the input MIDI through the same player"
                    : "This bundle has no input-MIDI rendering (rebuild it)"}
                  onClick={() => switchAudioMode("midi")}
                >
                  MIDI
                </button>
              </div>
              <div className="ofv-mode" role="group" aria-label="Choose predicted or actual tab">
                <button
                  type="button"
                  className={activeTabMode === "predicted" && audioMode === "tab" ? "active" : ""}
                  aria-pressed={activeTabMode === "predicted"}
                  title={audioMode === "midi" ? "Show the predicted tab (switches listening back to Tab)" : "Show the predicted tab"}
                  onClick={() => switchTabMode("predicted")}
                >
                  Predicted
                </button>
                <button
                  type="button"
                  className={activeTabMode === "actual" && audioMode === "tab" ? "active" : ""}
                  aria-pressed={activeTabMode === "actual"}
                  disabled={!hasReference}
                  title={audioMode === "midi"
                    ? "Show the validation reference fingering (switches listening back to Tab)"
                    : (hasReference ? "Show the validation reference fingering" : "No reference tab was saved for this sample")}
                  onClick={() => switchTabMode("actual")}
                >
                  Actual
                </button>
              </div>
              <div className="ofv-zoom">
                <button onClick={() => applyZoom(zoom - 0.1)} disabled={!ready.tab || zoom <= 0.5}
                  title="Zoom out">A−</button>
                <button className="ofv-zoom-label" onClick={() => applyZoom(1)}
                  disabled={!ready.tab} title="Reset zoom to 100%">
                  {Math.round(zoom * 100)}%</button>
                <button onClick={() => applyZoom(zoom + 0.1)} disabled={!ready.tab || zoom >= 2}
                  title="Zoom in">A+</button>
              </div>
            </div>
          </div>
          <div ref={tabScrollRef} className="ofv-tab-scroll"
            title="Drag the bottom edge to resize">
            {!ready.tab && <div className="ofv-skeleton">Rendering tab…</div>}
            <div ref={tabRenderRef} className="ofv-tab-render" />
          </div>
        </div>
      </div>
      <TokenInspector title="Input MIDI tokens" tokens={bundle.input_tokens} />
      <TokenInspector title="Predicted TAB tokens" tokens={bundle.tab_tokens} />
      {hasReference && (
        <TokenInspector title="Actual reference TAB tokens" tokens={bundle.reference_tab_tokens} />
      )}
      <div className="ofv-foot">
        One AlphaTab transport drives the tab cursor, the seek bar and the
        piano-roll playhead — tab and roll stay in sync. Flip Listen between
        Tab and MIDI to A/B the fingering against the input through the same
        player; the green bar tracks the sounding measures.
        {hasReference && " Switch Predicted/Actual above the score to compare fingerings."}
      </div>
    </div>
  );
}

export function mount(el, bundle) {
  const root = createRoot(el);
  root.render(React.createElement(TabDataViewer, { bundle }));
  return root;
}
