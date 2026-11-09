import React from "react";
import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const SOUNDFONT_URL =
  "https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/soundfont/sonivox.sf2";
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
.ofv-midi-body midi-player { width: 100%; }
.ofv-piano-roll { width: 100%; height: 280px; margin-top: 8px; border: 1px solid #e2e8f0;
  border-radius: 8px; overflow: hidden; background: #fff; }
@media (max-width: 760px) { .ofv-piano-roll { height: 210px; } }
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
.ofv-skeleton { padding: 24px; color: #94a3b8; font-size: 13px; text-align: center;
  background: repeating-linear-gradient(45deg, #f8fafc, #f8fafc 10px, #f1f5f9 10px, #f1f5f9 20px); }
.ofv-tokens { margin-top: 10px; }
.ofv-tokens summary { cursor: pointer; font-size: 13px; color: #475569; }
.ofv-tokens pre { white-space: pre-wrap; word-break: break-word; background: #f8fafc;
  border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px; max-height: 160px;
  overflow: auto; font-size: 12px; }
.ofv-foot { margin-top: 8px; font-size: 12px; color: #94a3b8; }
`;

function b64ToBlobUrl(b64, mime) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

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
  const midiWrapRef = useRef(null);
  const tabScrollRef = useRef(null);
  const tabRenderRef = useRef(null);
  const apiRef = useRef(null);
  const playerRef = useRef(null);
  const rollRef = useRef(null);
  const trackRef = useRef([]);
  const [ready, setReady] = useState({ midi: false, tab: false });
  const [tabPlaying, setTabPlaying] = useState(false);
  const [tabMode, setTabMode] = useState("predicted");
  const pendingTickRef = useRef(null);
  const hasReference = Boolean(bundle.reference_alpha_tex);
  const activeTabMode = tabMode === "actual" && hasReference ? "actual" : "predicted";
  const activeTex = activeTabMode === "actual" ? bundle.reference_alpha_tex : bundle.alpha_tex;
  const [muted, setMuted] = useState(false);
  const [tempo, setTempo] = useState(bundle.tempo || 120);
  const [tick, setTick] = useState(null);
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

  // MIDI panel: html-midi-player supplies compact audio controls; WaveRoll
  // (createPianoRoll) draws the grid, notes and a scrolling playhead. The
  // two are synchronized through the player's start/stop events and seek bar.
  useEffect(() => {
    const host = midiWrapRef.current;
    if (!host) return;
    host.innerHTML = "";
    let url = null;
    let roll = null;
    let rollFrame = null;
    let resizeObserver = null;
    let seekInput = null;
    let seekChange = null;
    let cancelled = false;
    try {
      url = b64ToBlobUrl(bundle.midi_b64, "audio/midi");
      const player = document.createElement("midi-player");
      player.setAttribute("src", url);
      player.setAttribute("sound-font", "");
      const rollHost = document.createElement("div");
      rollHost.className = "ofv-piano-roll";
      rollHost.setAttribute("role", "img");
      rollHost.setAttribute("aria-label", "Input MIDI piano roll");
      host.appendChild(player);
      host.appendChild(rollHost);
      playerRef.current = player;

      const drawPlayhead = () => {
        rollFrame = null;
        if (!roll || !player) return;
        try {
          roll.setTime(player.currentTime || 0);
        } catch (_) {}
      };
      const startFollow = () => {
        if (rollFrame != null) return;
        const tick = () => {
          rollFrame = null;
          drawPlayhead();
          if (player.playing && !cancelled) {
            rollFrame = window.requestAnimationFrame(tick);
          }
        };
        rollFrame = window.requestAnimationFrame(tick);
      };
      const stopFollow = () => {
        if (rollFrame != null) {
          window.cancelAnimationFrame(rollFrame);
          rollFrame = null;
        }
        drawPlayhead();
      };
      player.addEventListener("start", startFollow);
      player.addEventListener("loop", startFollow);
      player.addEventListener("stop", stopFollow);

      player.addEventListener("load", () => {
        setReady((r) => ({ ...r, midi: true }));
        drawPlayhead();
      });

      // The seek bar lives in the player's open shadow root; no public
      // seeking event exists, so mirror its input/change events.
      const seekBar = player.shadowRoot && player.shadowRoot.querySelector(".seek-bar");
      if (seekBar) {
        seekInput = () => drawPlayhead();
        seekChange = () => drawPlayhead();
        seekBar.addEventListener("input", seekInput);
        seekBar.addEventListener("change", seekChange);
      }

      if (window.WaveRoll && window.WaveRoll.createPianoRoll) {
        const notes = rollNotes(bundle);
        window.WaveRoll.createPianoRoll(
          rollHost,
          notes,
          pianoRollOptions(notes)
        ).then((instance) => {
          if (cancelled) {
            try { instance.destroy(); } catch (_) {}
            return;
          }
          roll = instance;
          rollRef.current = instance;
          try { rollHost._ofvRoll = instance; } catch (_) {}
          drawPlayhead();
          resizeObserver = new ResizeObserver(() => {
            try {
              instance.resize(rollHost.clientWidth, rollHost.clientHeight);
            } catch (_) {}
          });
          resizeObserver.observe(rollHost);
        }).catch((err) => {
          addWarn(`Piano roll failed: ${err}`);
          console.error(err);
        });
      } else {
        addWarn("WaveRoll failed to load (CDN blocked?). Piano roll disabled.");
      }

      if (player.start === undefined && !customElements.get("midi-player")) {
        addWarn("html-midi-player failed to load (CDN blocked?). MIDI panel disabled.");
      }
    } catch (err) {
      addWarn(`MIDI panel failed: ${err}`);
      console.error(err);
    }
    return () => {
      cancelled = true;
      if (rollFrame != null) window.cancelAnimationFrame(rollFrame);
      if (resizeObserver) resizeObserver.disconnect();
      if (seekInput) {
        const bar = playerRef.current && playerRef.current.shadowRoot &&
          playerRef.current.shadowRoot.querySelector(".seek-bar");
        if (bar) bar.removeEventListener("input", seekInput);
      }
      if (seekChange) {
        const bar = playerRef.current && playerRef.current.shadowRoot &&
          playerRef.current.shadowRoot.querySelector(".seek-bar");
        if (bar) bar.removeEventListener("change", seekChange);
      }
      try {
        roll && roll.destroy && roll.destroy();
      } catch (_) {}
      rollRef.current = null;
      try {
        playerRef.current && playerRef.current.stop && playerRef.current.stop();
      } catch (_) {}
      if (url) URL.revokeObjectURL(url);
      host.innerHTML = "";
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bundle.midi_b64, bundle.title]);

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
      api = new AT.AlphaTabApi(el, {
        player: {
          soundFont: SOUNDFONT_URL,
          enablePlayer: true,
          enableCursor: true,
          scrollElement: tabScrollRef.current || el,
        },
      });
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
                api.tickPosition = beat.playbackStart;
                setTick(beat.playbackStart);
              }
            } catch (_) {}
          });
        }
      } catch (_) {}
      try {
        if (api.playedBeatChanged && api.playedBeatChanged.on) {
          api.playedBeatChanged.on(() => {
            try {
              if (typeof api.tickPosition === "number") setTick(api.tickPosition);
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

  // Swap the score when the user toggles predicted/actual, keeping the same
  // player so playback/seeking survives the switch.
  useEffect(() => {
    const api = apiRef.current;
    if (!api) return;
    try {
      if (api.settings && api.settings.display) {
        api.settings.display.scale = zoomRef.current;
      }
      api.tex(activeTex);
    } catch (err) {
      addWarn(`AlphaTab failed to render this score: ${err}`);
      return;
    }

    const target = pendingTickRef.current;
    if (target == null) return;

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
            setTick(target);
            pendingTickRef.current = null;
            stop();
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
    giveUpTimer = setTimeout(() => {
      pendingTickRef.current = null;
      stop();
    }, 15000);
    return () => {
      stop();
      for (const name of ["scoreLoaded", "renderFinished", "playerReady"]) {
        unlisten(name, verify);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTex, bundle.title]);

  const playTab = () => {
    const api = apiRef.current;
    if (!api) return;
    try {
      if (typeof api.playPause === "function") api.playPause();
      else if (typeof api.play === "function") api.play();
      else {
        addWarn("This AlphaTab build exposes no play method.");
        return;
      }
      setTabPlaying(true);
    } catch (err) {
      addWarn(`Tab playback failed: ${err}`);
    }
  };
  const stopTab = () => {
    try {
      apiRef.current && apiRef.current.stop && apiRef.current.stop();
    } catch (_) {}
    setTabPlaying(false);
  };
  const switchTabMode = (mode) => {
    if (mode === activeTabMode || (mode === "actual" && !hasReference)) return;
    try {
      pendingTickRef.current =
        apiRef.current && typeof apiRef.current.tickPosition === "number"
          ? apiRef.current.tickPosition
          : tick;
    } catch (_) {
      pendingTickRef.current = tick;
    }
    try {
      apiRef.current && apiRef.current.stop && apiRef.current.stop();
    } catch (_) {}
    setTabPlaying(false);
    setReady((r) => ({ ...r, tab: false }));
    setTabMode(mode);
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
  const playMidi = () => {
    try {
      playerRef.current && playerRef.current.start && playerRef.current.start();
    } catch (err) {
      addWarn(`MIDI playback failed: ${err}`);
    }
  };
  const stopMidi = () => {
    try {
      playerRef.current && playerRef.current.stop && playerRef.current.stop();
    } catch (_) {}
  };
  const playBoth = () => {
    playMidi();
    try {
      apiRef.current && apiRef.current.play && apiRef.current.play();
      setTabPlaying(true);
    } catch (_) {
      playTab();
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
        <button className="ofv-btn primary" onClick={playMidi} disabled={!ready.midi}>
          ▶ MIDI input
        </button>
        <button className="ofv-btn" onClick={stopMidi} title="Stop MIDI">
          ⏹
        </button>
        <button className="ofv-btn primary" onClick={playTab} disabled={!ready.tab}>
          {tabPlaying ? "⏸ Tab" : "▶ Tab"}
        </button>
        <button className="ofv-btn" onClick={stopTab} title="Stop tab">
          ⏹
        </button>
        <button
          className="ofv-btn"
          onClick={toggleMute}
          disabled={!ready.tab}
          title="Mute/unmute the tab track for A/B listening"
        >
          {muted ? "🔈 Unmute tab" : "🔇 Mute tab"}
        </button>
        <button
          className="ofv-btn"
          onClick={playBoth}
          disabled={!(ready.midi && ready.tab)}
          title="Start both players together (approximate sync)"
        >
          ▶▶ Both
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
        {tick !== null && tick !== undefined && <span className="ofv-tick">tick {tick}</span>}
      </div>
      <div className="ofv-grid">
        <div className="ofv-card">
          <h4>Input MIDI — piano roll (play above; wheel/pinch zooms)</h4>
          <div className="ofv-midi-body">
            {!ready.midi && <div className="ofv-skeleton">Loading MIDI player…</div>}
            <div ref={midiWrapRef} data-uid={uid} />
          </div>
        </div>
        <div className="ofv-card">
          <div className="ofv-card-head">
            <h4>{activeTabMode === "actual" ? "Actual reference tab" : "Predicted tab"} — click the score to seek</h4>
            <div className="ofv-tab-controls">
              <div className="ofv-mode" role="group" aria-label="Choose predicted or actual tab">
                <button
                  type="button"
                  className={activeTabMode === "predicted" ? "active" : ""}
                  aria-pressed={activeTabMode === "predicted"}
                  onClick={() => switchTabMode("predicted")}
                >
                  Predicted
                </button>
                <button
                  type="button"
                  className={activeTabMode === "actual" ? "active" : ""}
                  aria-pressed={activeTabMode === "actual"}
                  disabled={!hasReference}
                  title={hasReference ? "Show the validation reference fingering" : "No reference tab was saved for this sample"}
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
        Tab audio uses the Salamander soundfont via CDN; MIDI panel uses the
        Magenta soundfont. Panels play independently — “Both” starts them together.
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
