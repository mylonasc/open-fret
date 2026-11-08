import React from "react";
import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const SOUNDFONT_URL =
  "https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/soundfont/sonivox.sf2";

function b64ToBlobUrl(b64, mime) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

function TokenInspector({ title, tokens }) {
  const text = Array.isArray(tokens) ? tokens.join(" ") : String(tokens ?? "");
  return (
    <details style={{ marginTop: 8 }}>
      <summary>
        {title} ({Array.isArray(tokens) ? tokens.length : 0} tokens)
      </summary>
      <pre
        style={{
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          background: "#f6f6f6",
          padding: 8,
          maxHeight: 160,
          overflow: "auto",
          fontSize: 12,
        }}
      >
        {text}
      </pre>
    </details>
  );
}

export function TabDataViewer({ bundle }) {
  const midiWrapRef = useRef(null);
  const tabRef = useRef(null);
  const apiRef = useRef(null);
  const playerRef = useRef(null);
  const trackRef = useRef([]);
  const [muted, setMuted] = useState(false);
  const [ready, setReady] = useState({ midi: false, tab: false });
  const [tabPlaying, setTabPlaying] = useState(false);
  const [tempo, setTempo] = useState(bundle.tempo || 120);
  const [tick, setTick] = useState(null);
  const [warn, setWarn] = useState([]);

  const addWarn = (m) => setWarn((w) => (w.includes(m) ? w : [...w, m]));

  // --- MIDI panel: html-midi-player custom elements (imperative: works on React 18/19) ---
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
      player.setAttribute("visualizer", `#ofv-midi-viz-${host.dataset.uid}`);
      const viz = document.createElement("midi-visualizer");
      viz.setAttribute("type", "piano-roll");
      viz.id = `ofv-midi-viz-${host.dataset.uid}`;
      host.appendChild(player);
      host.appendChild(viz);
      playerRef.current = player;
      player.addEventListener("load", () => setReady((r) => ({ ...r, midi: true })));
      // If the custom elements were never registered (CDN blocked), flag it.
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
  }, [bundle.midi_b64, bundle.title]);

  // --- TAB panel: AlphaTab score with player ---
  // NOTE: settings.tex is a *boolean*; tex CONTENT loads via api.tex().
  useEffect(() => {
    const el = tabRef.current;
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
          scrollElement: el,
        },
      });
      apiRef.current = api;
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
    } catch (_) {
      playTab();
    }
  };
  const onTempo = (v) => {
    const t = Number(v);
    setTempo(t);
    try {
      const api = apiRef.current;
      // playbackSpeed exists on recent AlphaTab builds; guard otherwise.
      if (api && "playbackSpeed" in api) api.playbackSpeed = t / (bundle.tempo || 120);
      else addWarn("Tempo scaling needs AlphaTab playbackSpeed (not in this build).");
    } catch (_) {}
  };

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 960 }}>
      <h3 style={{ margin: "4px 0" }}>{bundle.title || "open-fret sample"}</h3>
      {warn.length > 0 && (
        <div style={{ background: "#fff4e5", padding: 8, marginBottom: 8 }}>
          {warn.map((w, i) => (
            <div key={i}>⚠️ {w}</div>
          ))}
        </div>
      )}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        <button onClick={playMidi} disabled={!ready.midi}>
          ▶ MIDI input
        </button>
        <button onClick={stopMidi}>⏹ MIDI</button>
        <button onClick={playTab} disabled={!ready.tab}>
          {tabPlaying ? "⏸ Tab" : "▶ Tab"}
        </button>
        <button onClick={stopTab}>⏹ Tab</button>
        <button onClick={toggleMute} disabled={!ready.tab}>
          {muted ? "🔈 Unmute tab" : "🔇 Mute tab"}
        </button>
        <button onClick={playBoth} disabled={!(ready.midi && ready.tab)}>
          ▶▶ Both
        </button>
        <label style={{ alignSelf: "center" }}>
          tempo{" "}
          <select value={tempo} onChange={(e) => onTempo(e.target.value)}>
            {[60, 80, 100, 120, 140, 160].map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        {tick !== null && tick !== undefined && (
          <span style={{ alignSelf: "center", color: "#555" }}>tick {tick}</span>
        )}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div>
          <h4 style={{ margin: "4px 0" }}>
            Input MIDI ({bundle.meta ? bundle.meta.n_input_notes : "?"} notes)
          </h4>
          <div ref={midiWrapRef} data-uid={bundle._uid || "midi"} />
        </div>
        <div>
          <h4 style={{ margin: "4px 0" }}>Predicted tab (AlphaTab — click score to seek)</h4>
          <div
            ref={tabRef}
            style={{ maxHeight: 420, overflow: "auto", border: "1px solid #ddd" }}
          />
        </div>
      </div>
      <TokenInspector title="Input MIDI tokens" tokens={bundle.input_tokens} />
      <TokenInspector title="Predicted TAB tokens" tokens={bundle.tab_tokens} />
    </div>
  );
}

export function mount(el, bundle) {
  const root = createRoot(el);
  root.render(React.createElement(TabDataViewer, { bundle }));
  return root;
}
