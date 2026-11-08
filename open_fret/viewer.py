"""Viewer data plumbing for ui/tab_data_viewer (Issue #13).

Builds a self-contained bundle from a model I/O pair:
  input  = MIDI notes (file or reconstructed from NO_/NF_/TS_ tokens)
  output = predicted TAB tokens rendered to AlphaTex

``to_html`` wraps the bundle with the React viewer (dist bundle inlined)
so a notebook cell can render playable + seekable tab + MIDI.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.render import TabRenderer

TICKS_PER_BEAT = 480


def renderer_for_input(midi_tokens, tempo=120):
    """Match generated tab audio to the source window's actual tuning/capo."""
    if isinstance(midi_tokens, str):
        midi_tokens = midi_tokens.split()
    tuning, capo = None, 0
    for token in midi_tokens[:2]:
        if token.startswith("TUNING_"):
            tuning = [int(v) for v in token[7:].split("_")]
        elif token.startswith("CAPO_"):
            capo = int(token[5:])
    if tuning is None:
        return TabRenderer(tempo=tempo)
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    while tuning and tuning[-1] < 0:
        tuning.pop()
    if any(p < 0 for p in tuning):
        raise ValueError("Viewer cannot omit an internal unavailable string")
    written = " ".join(f"{names[(p + capo) % 12]}{(p + capo) // 12}" for p in tuning)
    return TabRenderer(tempo=tempo, tuning_str=written)


def midi_tokens_to_notes(midi_tokens, default_duration=480):
    """Pair ON/OFF events into notes with absolute ticks.

    Returns list of {"start", "end", "pitch"} sorted by start time.
    Stray OFFs are ignored; dangling ONs close at track end.
    """
    if isinstance(midi_tokens, str):
        midi_tokens = midi_tokens.split()
    time = 0
    open_notes: dict[int, list[int]] = {}
    notes: list[dict] = []
    for tok in midi_tokens:
        if tok.startswith("TS_"):
            try:
                time += int(tok.split("_")[1])
            except ValueError:
                continue
        elif tok.startswith("NO_"):
            try:
                p = int(tok.split("_")[1])
            except ValueError:
                continue
            open_notes.setdefault(p, []).append(time)
        elif tok.startswith("NF_"):
            try:
                p = int(tok.split("_")[1])
            except ValueError:
                continue
            stack = open_notes.get(p)
            if stack:
                notes.append({"start": stack.pop(), "end": time, "pitch": p})
    for p, stack in open_notes.items():
        for s in stack:
            end = max(time, s + default_duration)
            notes.append({"start": s, "end": end, "pitch": p})
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def notes_to_midi_bytes(notes, tempo=120, ticks_per_beat=TICKS_PER_BEAT):
    """Serialize notes to Standard MIDI bytes (pretty_midi).

    Program is Acoustic Guitar (nylon) so the html-midi-player panel — which
    picks its soundfont samples from the MIDI program — sounds classical,
    matching the AlphaTab tab track (``\\instrument 24``).
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    program = pretty_midi.instrument_name_to_program("Acoustic Guitar (nylon)")
    instrument = pretty_midi.Instrument(program=program, is_drum=False)
    sec_per_tick = 60.0 / (tempo * ticks_per_beat)
    for n in notes:
        start = max(0.0, n["start"] * sec_per_tick)
        end = max(start + 0.01, n["end"] * sec_per_tick)
        instrument.notes.append(pretty_midi.Note(
            velocity=100, pitch=int(n["pitch"]), start=start, end=end))
    instrument.notes.sort(key=lambda x: x.start)
    pm.instruments.append(instrument)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pm.write(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def build_view_bundle(midi_path=None, midi_tokens=None, tab_tokens=None,
                      title="open-fret sample", tempo=120):
    """Build a viewer bundle dict (JSON-serializable).

    Provide either ``midi_path`` (a .mid file) or ``midi_tokens`` (model
    input token stream). ``tab_tokens`` is the predicted TAB token stream.
    """
    from open_fret.midi_io import midi_file_to_tokens

    if tab_tokens is None:
        raise ValueError("tab_tokens is required (predicted TAB token stream).")
    if isinstance(tab_tokens, str):
        tab_tokens = tab_tokens.split()
    if midi_path is not None:
        with open(midi_path, "rb") as f:
            midi_bytes = f.read()
        input_tokens = midi_file_to_tokens(midi_path)
    elif midi_tokens is not None:
        if isinstance(midi_tokens, str):
            midi_tokens = midi_tokens.split()
        input_tokens = list(midi_tokens)
        notes = midi_tokens_to_notes(input_tokens)
        midi_bytes = notes_to_midi_bytes(notes, tempo=tempo)
    else:
        raise ValueError("Provide midi_path or midi_tokens.")

    alpha_tex = renderer_for_input(input_tokens, tempo=tempo).to_alphatex(tab_tokens, title=title)
    pitches = [n["pitch"] for n in midi_tokens_to_notes(input_tokens)]
    return {
        "title": title,
        "tempo": tempo,
        "ticks_per_beat": TICKS_PER_BEAT,
        "midi_b64": base64.b64encode(midi_bytes).decode("ascii"),
        "alpha_tex": alpha_tex,
        "input_tokens": input_tokens,
        "tab_tokens": list(tab_tokens),
        "meta": {
            "n_input_tokens": len(input_tokens),
            "n_tab_tokens": len(tab_tokens),
            "n_input_notes": len(midi_tokens_to_notes(input_tokens)),
            "pitch_min": min(pitches) if pitches else None,
            "pitch_max": max(pitches) if pitches else None,
        },
    }


ALPHATAB_JS = ("https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0"
               "/dist/alphaTab.min.js")
MIDI_PLAYER_JS = ("https://cdn.jsdelivr.net/combine/npm/tone@14.7.77,"
                  "npm/@magenta/music@1.23.0/es6/core.js,"
                  "npm/focus-visible@5,npm/html-midi-player@1.5.0")


def to_html(bundle, dist_js=None, mount_name="TabDataViewer", with_deps=True):
    """Render a self-contained viewer HTML string for notebook/page embedding.

    ``dist_js`` is the built UMD bundle text (reads
    ui/tab_data_viewer/dist/* at call sites). Each call mints unique element
    ids so multiple cells can coexist.

    Pass ``with_deps=False`` (and ``dist_js=None``) for the 2nd..Nth bundle
    on a page that already includes the CDN scripts + dist bundle once —
    re-including them redefines the ``midi-player`` custom element and
    throws. See ``scripts/make_run_viewer.py``.
    """
    uid = uuid.uuid4().hex[:8]
    # json.dumps never emits "</script>", but TAB/AlphaTex text is
    # user-derived; neutralize the closing tag unconditionally.
    bundle_json = json.dumps(bundle).replace("</", "<\\/")
    # NOTE: the dist bundle must come BEFORE the mount script — scripts
    # execute in order, and mounting needs window[mount_name] to exist.
    deps = ""
    if with_deps:
        if dist_js is None:
            raise ValueError("dist_js is required when with_deps=True.")
        deps = (f'<script src="{MIDI_PLAYER_JS}"></script>\n'
                f'<script src="{ALPHATAB_JS}"></script>\n'
                f"<script>\n{dist_js}\n</script>\n")
    return f"""{deps}<div id="ofv-{uid}" class="ofv-root"></div>
<script>
(function() {{
  if (!window.{mount_name}) {{
    document.getElementById("ofv-{uid}").innerHTML =
      "<p><b>Viewer bundle failed to load.</b> "
      + "The component script (dist) may be missing; tokens are still "
      + "inspectable in the notebook variables.</p>";
    return;
  }}
  window.__OFV_BUNDLES = window.__OFV_BUNDLES || {{}};
  window.__OFV_BUNDLES["{uid}"] = {bundle_json};
  try {{
    window.{mount_name}.mount(
      document.getElementById("ofv-{uid}"),
      window.__OFV_BUNDLES["{uid}"]);
  }} catch (err) {{
    document.getElementById("ofv-{uid}").innerHTML =
      "<p><b>Viewer failed to start:</b> " + String(err) + "</p>";
    console.error(err);
  }}
}})();
</script>"""


def show(bundle_or_path, dist_path=None):
    """Display a bundle in a Jupyter cell. Returns an IPython HTML object."""
    from IPython.display import HTML
    if isinstance(bundle_or_path, str):
        with open(bundle_or_path, encoding="utf-8") as f:
            bundle = json.load(f)
    else:
        bundle = bundle_or_path
    if dist_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        dist_dir = os.path.join(os.path.dirname(here), "ui", "tab_data_viewer", "dist")
        cands = sorted(f for f in os.listdir(dist_dir) if f.endswith(".js"))
        if not cands:
            raise FileNotFoundError(f"No built viewer bundle in {dist_dir}. Run npm run build.")
        dist_path = os.path.join(dist_dir, cands[0])
    with open(dist_path, encoding="utf-8") as f:
        dist_js = f.read()
    return HTML(to_html(bundle, dist_js))
