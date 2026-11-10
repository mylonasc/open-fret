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
import html as html_utils
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_fret.viewer import (ALPHATAB_JS, WAVE_ROLL_JS, build_view_bundle,
                               to_html)


def build_run_viewer(run_dir, step=None, step_dir=None, out_name="viewer.html"):
    args = SimpleNamespace(run_dir=run_dir, step=step, step_dir=step_dir, out_name=out_name)
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
    metric_summary = html_utils.escape(json.dumps(manifest.get("eval_metrics") or {}, indent=2))

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "ui", "tab_data_viewer", "dist",
                            "tab-data-viewer.umd.js"), encoding="utf-8") as f:
        dist_js = f.read()

    sections = []
    n_ok = 0
    for song in manifest["songs"]:
        if "error" in song:
            sections.append(f"<p>⚠️ {html_utils.escape(song['name'])}: {html_utils.escape(song['error'])}</p>")
            continue
        base = song["file"]
        with open(os.path.join(sdir, base + ".input.txt"),
                  encoding="utf-8") as f:
            midi_tokens = f.read().split()
        with open(os.path.join(sdir, base + ".pred.txt"),
                  encoding="utf-8") as f:
            pred_tokens = f.read().split()
        ref_path = os.path.join(sdir, base + ".ref.txt")
        ref_tokens = None
        if os.path.isfile(ref_path):
            with open(ref_path, encoding="utf-8") as f:
                ref_tokens = f.read().split() or None
        title = f"{song['name']} — predicted vs actual @ {step}"
        try:
            bundle = build_view_bundle(midi_tokens=midi_tokens,
                                       tab_tokens=pred_tokens,
                                       reference_tab_tokens=ref_tokens,
                                       title=title, tempo=song.get("tempo", 120))
            sections.append(f"<h2>{html_utils.escape(title)}</h2>\n"
                            f"<p>pred {len(pred_tokens)} tokens, "
                            f"ref {song.get('n_ref')} tokens</p>\n"
                            + to_html(bundle, with_deps=False))
        except Exception as e:  # noqa: BLE001 - one bad song (e.g. an
            # unrenderable same-onset conflict in pred or ref) must never
            # fail the viewer, let alone kill a training run.
            sections.append(f"<p>⚠️ {html_utils.escape(song['name'])}: viewer skipped "
                            f"({html_utils.escape(type(e).__name__)}: "
                            f"{html_utils.escape(str(e)[:200])})</p>")
            continue
        n_ok += 1

    loss_str = f"{eval_loss:.4f}" if eval_loss is not None else "n/a"
    deps = (f'<script src="{WAVE_ROLL_JS}"></script>\n'
            f'<script src="{ALPHATAB_JS}"></script>\n'
            f"<script>\n{dist_js}\n</script>\n")
    html = ("<!doctype html>\n<html><head><meta charset=\"utf-8\">"
            "<link rel=\"icon\" href=\"data:,\">"
            f"<title>open-fret run {manifest.get('run')} {step}</title>"
            "</head><body>\n"
            f"<h1>open-fret: {manifest.get('run')} @ {step} "
             f"(eval loss {loss_str})</h1>\n"
             f"<details><summary>Loss components / validation metrics</summary><pre>{metric_summary}</pre></details>"
            "<p>Needs internet access for CDN scripts + soundfont. "
            "Each section: piano roll (left) vs tab (right) on one AlphaTab "
            "transport — Play/Pause, the seek bar, score clicks and roll "
            "clicks move both views together; use the "
            "Predicted/Actual toggle above the score to compare fingerings.</p>\n"
            + deps
            + "\n<hr>\n".join(sections) + "\n</body></html>\n")
    out = os.path.join(sdir, args.out_name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out} ({os.path.getsize(out)} bytes, {n_ok} songs)")
    # Discoverable history without rebuilding earlier pages.
    links = []
    for name in sorted(os.listdir(samples)):
        path = os.path.join(samples, name, "manifest.json")
        if not os.path.isfile(path) or not os.path.isfile(os.path.join(samples, name, "viewer.html")):
            continue
        with open(path, encoding="utf-8") as f:
            entry = json.load(f)
        values = entry.get("eval_metrics") or {}
        links.append(f'<li><a href="{html_utils.escape(name)}/viewer.html">{html_utils.escape(name)}</a> '
                     f'{html_utils.escape(json.dumps(values))}</li>')
    with open(os.path.join(samples, "index.html"), "w", encoding="utf-8") as f:
        f.write('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">'
                '<h1>Validation / checkpoint viewers</h1><ul>'
                + "\n".join(links) + "</ul>")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--step", type=int, default=None, help="Default: latest step folder.")
    p.add_argument("--step-dir", default=None, help="Exact samples subfolder; overrides --step.")
    p.add_argument("--out-name", default="viewer.html")
    args = p.parse_args()
    build_run_viewer(**vars(args))


if __name__ == "__main__":
    main()
