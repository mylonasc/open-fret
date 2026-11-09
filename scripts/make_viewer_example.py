"""Generate the tab_data_viewer example output (Issue #13).

Picks a real processed window (standard tuning, chord-heavy, 100-300 MIDI
tokens), builds a viewer bundle with ground-truth TAB shown as the
"predicted" track (labeled as such), and writes:
  ui/tab_data_viewer/demo/bundle.example.json
  ui/tab_data_viewer/demo/example_output.html  (self-contained page)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.viewer import build_view_bundle, to_html


def pick_window(indiv_dir, lo=100, hi=300):
    best = None
    best_score = -1
    checked = 0
    for name in sorted(os.listdir(indiv_dir)):
        if not name.endswith(".json") or name.startswith("_"):
            continue
        checked += 1
        if checked > 4000:
            break
        with open(os.path.join(indiv_dir, name), encoding="utf-8") as f:
            d = json.load(f)
        if d.get("tuning") != [64, 59, 55, 50, 45, 40]:
            continue
        m, t = len(d["midi"]), len(d["tab"])
        if not (lo <= m <= hi):
            continue
        n_tab = sum(1 for x in d["tab"] if x.startswith("TAB_"))
        n_ts = sum(1 for x in d["tab"] if x.startswith("TS_"))
        density = n_tab / max(n_ts, 1)  # >1 means chords
        if density > best_score and n_tab >= 20:
            best_score = density
            best = (name, d)
    if best is None:
        raise RuntimeError("No suitable example window found.")
    return best


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    indiv = os.path.join(root, "dataset", "processed", "individual")
    demo = os.path.join(root, "ui", "tab_data_viewer", "demo")
    os.makedirs(demo, exist_ok=True)
    name, d = pick_window(indiv)
    print(f"Example window: {name}")
    print(f"  midi tokens: {len(d['midi'])}, tab tokens: {len(d['tab'])}")
    bundle = build_view_bundle(
        midi_tokens=d["midi"], tab_tokens=d["tab"],
        title=f"open-fret viewer example ({d.get('source', name)})",
        tempo=120)
    # Honest labeling: ground-truth tab stands in for a model prediction.
    bundle["title"] += " — ground-truth tab shown as predicted (demo)"
    with open(os.path.join(demo, "bundle.example.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f)
    dist = os.path.join(root, "ui", "tab_data_viewer", "dist",
                        "tab-data-viewer.umd.js")
    with open(dist, encoding="utf-8") as f:
        dist_js = f.read()
    html = ("<!doctype html>\n<html><head><meta charset=\"utf-8\">"
            "<link rel=\"icon\" href=\"data:,\">"
            "<title>open-fret viewer example</title></head><body>\n"
            "<h2>open-fret viewer example (self-contained)</h2>\n"
            "<p>Needs internet access for CDN scripts + soundfont. "
            "Left: piano roll. Right: predicted tab (ground truth shown for demo). "
            "One AlphaTab transport drives both views in sync.</p>\n"
            + to_html(bundle, dist_js) + "\n</body></html>\n")
    out = os.path.join(demo, "example_output.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
