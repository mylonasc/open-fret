import React from "react";
import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const SOUNDFONT_URL =
  "https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/soundfont/sonivox.sf2";

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
@media (max-width: 760px) { .ofv-grid { grid-template-columns: 1fr; } }
.ofv-card { border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; background: #fff;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); min-width: 0; }
.ofv-card h4 { margin: 0; padding: 8px 12px; font-size: 13px; font-weight: 650;
  background: #f8fafc; border-bottom: 1px solid #e2e8f0; }
.ofv-midi-body { padding: 10px 12px; min-height: 120px; }
.ofv-midi-body midi-player { width: 100%; }
.ofv-card-head { display: flex; align-items: center; justify-content: space-between;
  gap: 8px; padding: 6px 12px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }
.ofv-card-head h4 { margin: 0; padding: 0; background: none; border: none;
  font-size: 13px; font-weight: 650; }
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
  const trackRef = useRef([]);
  const [ready, setReady] = useState({ midi: false, tab: false });
  const [tabPlaying, setTabPlaying] = useState(false);
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

  // MIDI panel: html-midi-player custom elements (imperative: works on React 18/19)
  useEffect(() => {
    const host = midiWrapRef.current;
    if (!host) return;
    host.innerHTML = "";
    let url = null;
    try {
      url = b64ToBlobUrl(bundle.midi_b64, "audio/midi");
      const player = document.createElement("midi-player");
      player.setAttribute("src", url);
      player.setAttribute("sound-font", "");
      player.setAttribute("visualizer", `#ofv-midi-viz-${uid}`);
      const viz = document.createElement("midi-visualizer");
      viz.setAttribute("type", "piano-roll");
      viz.id = `ofv-midi-viz-${uid}`;
      host.appendChild(player);
      host.appendChild(viz);
      playerRef.current = player;
      player.addEventListener("load", () => setReady((r) => ({ ...r, midi: true })));
      if (player.start === undefined && !customElements.get("midi-player")) {
        addWarn("html-midi-player failed to load (CDN blocked?). MIDI panel disabled.");
      }
    } catch (err) {
      addWarn(`MIDI panel failed: ${err}`);
      console.error(err);
    }
    return () => {
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
      api.tex(bundle.alpha_tex);
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
  }, [bundle.alpha_tex, bundle.title]);

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
          {meta.n_tab_tokens != null && <span className="ofv-badge">🎸 {meta.n_tab_tokens} tab tokens out</span>}
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
          <h4>Input MIDI — piano roll (click play above, seek in the bar)</h4>
          <div className="ofv-midi-body">
            {!ready.midi && <div className="ofv-skeleton">Loading MIDI player…</div>}
            <div ref={midiWrapRef} data-uid={uid} />
          </div>
        </div>
        <div className="ofv-card">
          <div className="ofv-card-head">
            <h4>Predicted tab — click the score to seek</h4>
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
          <div ref={tabScrollRef} className="ofv-tab-scroll"
            title="Drag the bottom edge to resize">
            {!ready.tab && <div className="ofv-skeleton">Rendering tab…</div>}
            <div ref={tabRenderRef} className="ofv-tab-render" />
          </div>
        </div>
      </div>
      <TokenInspector title="Input MIDI tokens" tokens={bundle.input_tokens} />
      <TokenInspector title="Predicted TAB tokens" tokens={bundle.tab_tokens} />
      <div className="ofv-foot">
        Tab audio uses the Salamander soundfont via CDN; MIDI panel uses the
        Magenta soundfont. Panels play independently — “Both” starts them together.
      </div>
    </div>
  );
}

export function mount(el, bundle) {
  const root = createRoot(el);
  root.render(React.createElement(TabDataViewer, { bundle }));
  return root;
}
