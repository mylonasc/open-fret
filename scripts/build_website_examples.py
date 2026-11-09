"""Build per-song playable viewer pages for the Docusaurus website.

Reads real model predictions (input/pred/ref triples) from a training-run
samples step — by default the five fixed validation windows of the curated
run — and writes one self-contained page per song:
  website/static/examples/<slug>/player.html

Run:  python scripts/build_website_examples.py [--run-dir models/<run>]
      [--step-dir step-00011043]
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.viewer import build_view_bundle, to_html  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SONGS = [
    ("12_Stones__12_Stones_-_Crash.tex__t0__c0__w0",
     "crash", "12 Stones — Crash",
     "Driving eighth-note riff at 130 BPM; single-note lines with chord stabs."),
    ("12_Stones__12_Stones_-_The_Way_I_Feel.tex__t0__c0__w10",
     "the-way-i-feel", "12 Stones — The Way I Feel",
     "Mid-tempo 80 BPM verse window with mixed single notes and dyads."),
    ("1974_ad__pahilo_junema__1974_AD__Pahilo_Junema_-_Pahilo_Junima.tex__t0__c0__w0",
     "pahilo-junema", "1974 AD — Pahilo Junema",
     "Sparse 72 BPM melodic line with open strings."),
    ("1_God__1_God_-_Glory.tex__t0__c0__w0",
     "glory", "1 God — Glory",
     "Dense 133 BPM strummed chords; a workout for same-onset string assignment."),
    ("23_P.H__23_P.H._-_End_Games.tex__t0__c0__w0",
     "end-games", "23 P.H. — End Games",
     "120 BPM window balancing melody against accompaniment figures."),
]


def build_song(sdir, base, slug, title, blurb, dist_js, static_root):
    def read(suffix):
        with open(os.path.join(sdir, base + suffix), encoding="utf-8") as f:
            return f.read().split()

    midi_tokens = read(".input.txt")
    pred_tokens = read(".pred.txt")
    ref_path = os.path.join(sdir, base + ".ref.txt")
    ref_tokens = None
    if os.path.isfile(ref_path):
        with open(ref_path, encoding="utf-8") as f:
            ref_tokens = f.read().split() or None
    tempo = 120
    manifest_path = os.path.join(sdir, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        for song in manifest.get("songs", []):
            if song.get("file") == base and song.get("tempo"):
                tempo = int(song["tempo"])
    try:
        bundle = build_view_bundle(midi_tokens=midi_tokens, tab_tokens=pred_tokens,
                                   reference_tab_tokens=ref_tokens,
                                   title=f"{title} — model prediction vs reference",
                                   tempo=tempo)
    except ValueError as e:
        # A reference with a same-onset/same-string conflict cannot be
        # rendered faithfully; keep the prediction-only view and say so.
        print(f"WARNING {slug}: no reference ({e}); prediction only")
        ref_tokens = None
        bundle = build_view_bundle(midi_tokens=midi_tokens, tab_tokens=pred_tokens,
                                   title=f"{title} — model prediction",
                                   tempo=tempo)
    viewer = to_html(bundle, dist_js=dist_js)
    page = ("<!doctype html>\n<html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{title} — open-fret example</title></head><body>\n"
            + viewer + "\n</body></html>\n")
    out_dir = os.path.join(static_root, slug)
    os.makedirs(out_dir, exist_ok=True)
    # index.html (not player.html): folder URLs survive clean-URL redirects
    # both in `docusaurus serve` and on GitHub Pages.
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    meta = {"title": title, "blurb": blurb, "tempo": tempo,
            "n_input": len(midi_tokens), "n_pred": len(pred_tokens),
            "has_reference": ref_tokens is not None,
            "provenance": ("vldsavelyev/guitar_tab window; prediction by "
                           "full-curated-3ep-20261109; rights unresolved, see docs")}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="models/full-curated-3ep-20261109")
    ap.add_argument("--step-dir", default="step-00011043")
    ap.add_argument("--static-root",
                    default=os.path.join(ROOT, "website", "static", "examples"))
    args = ap.parse_args()
    sdir = os.path.join(ROOT, args.run_dir, "samples", args.step_dir)
    if not os.path.isdir(sdir):
        raise SystemExit(f"Not a directory: {sdir}")
    dist = os.path.join(ROOT, "ui", "tab_data_viewer", "dist",
                        "tab-data-viewer.umd.js")
    with open(dist, encoding="utf-8") as f:
        dist_js = f.read()
    for base, slug, title, blurb in SONGS:
        meta = build_song(sdir, base, slug, title, blurb, dist_js,
                          args.static_root)
        print(f"{slug}: {meta['n_input']} in / {meta['n_pred']} pred @ {meta['tempo']} BPM")


if __name__ == "__main__":
    main()
