"""Build a self-contained viewer HTML for one training-run sample step.

Reads <run>/samples/step-<N>/{*.input.txt,*.pred.txt,manifest.json} (written
by TabSampleCallback in scripts/train_model.py) and writes viewer.html next
to them: one playable MIDI + predicted-tab section per song.

Run:  python scripts/make_run_viewer.py [--run-dir models/<run>] [--step N]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.viewer import (ALPHATAB_JS, MIDI_PLAYER_JS, build_view_bundle,
                               to_html)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--step", type=int, default=None,
                   help="Default: latest step folder.")
    p.add_argument("--step-dir", default=None,
                   help="Use this exact samples subfolder (e.g. a -redecode "
                        "folder); overrides --step.")
    p.add_argument("--out-name", default="viewer.html")
    args = p.parse_args()

    samples = os.path.join(args.run_dir, "samples")
    if args.step_dir:
        step = args.step_dir
        sdir = os.path.join(samples, step)
        if not os.path.isdir(sdir):
            raise SystemExit(f"Not a directory: {sdir}")
    else:
        steps = sorted(d for d in os.listdir(samples)
                       if re.fullmatch(r"step-\d+", d))
        if not steps:
            raise SystemExit(f"No step folders in {samples}")
        if args.step is None:
            step = steps[-1]
        else:
            step = f"step-{args.step:08d}"
            if step not in steps:
                raise SystemExit(f"{step} not found; have: {steps}")
        sdir = os.path.join(samples, step)
    manifest = json.load(open(os.path.join(sdir, "manifest.json"),
                              encoding="utf-8"))
    eval_loss = (manifest.get("eval_metrics") or {}).get("eval_loss")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "ui", "tab_data_viewer", "dist",
                            "tab-data-viewer.umd.js"), encoding="utf-8") as f:
        dist_js = f.read()

    sections = []
    n_ok = 0
    for song in manifest["songs"]:
        if "error" in song:
            sections.append(f"<p>⚠️ {song['name']}: {song['error']}</p>")
            continue
        base = song["file"]
        with open(os.path.join(sdir, base + ".input.txt"),
                  encoding="utf-8") as f:
            midi_tokens = f.read().split()
        with open(os.path.join(sdir, base + ".pred.txt"),
                  encoding="utf-8") as f:
            pred_tokens = f.read().split()
        title = f"{song['name']} — predicted tab @ {step}"
        bundle = build_view_bundle(midi_tokens=midi_tokens,
                                   tab_tokens=pred_tokens, title=title)
        sections.append(f"<h2>{title}</h2>\n"
                        f"<p>pred {len(pred_tokens)} tokens, "
                        f"ref {song.get('n_ref')} tokens</p>\n"
                        + to_html(bundle, with_deps=False))
        n_ok += 1

    loss_str = f"{eval_loss:.4f}" if eval_loss is not None else "n/a"
    deps = (f'<script src="{MIDI_PLAYER_JS}"></script>\n'
            f'<script src="{ALPHATAB_JS}"></script>\n'
            f"<script>\n{dist_js}\n</script>\n")
    html = ("<!doctype html>\n<html><head><meta charset=\"utf-8\">"
            "<link rel=\"icon\" href=\"data:,\">"
            f"<title>open-fret run {manifest.get('run')} {step}</title>"
            "</head><body>\n"
            f"<h1>open-fret: {manifest.get('run')} @ {step} "
            f"(eval loss {loss_str})</h1>\n"
            "<p>Needs internet access for CDN scripts + soundfont. "
            "Each section: input MIDI (left) vs model-predicted tab (right).</p>\n"
            + deps
            + "\n<hr>\n".join(sections) + "\n</body></html>\n")
    out = os.path.join(sdir, args.out_name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out} ({os.path.getsize(out)} bytes, {n_ok} songs)")


if __name__ == "__main__":
    main()
