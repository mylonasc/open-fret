"""Export tab samples from a trained checkpoint with sane decoding.

Greedy decoding of the tiny T5 collapses into 1-2 token loops (never emits
EOS), which misrepresents what the model learned. This script decodes with
beam search + no-repeat-ngram guards (the same family of settings the HPO
loss curves reflect via teacher forcing) and optional neighbor-search
snapping, then writes the same folder layout as TabSampleCallback:

  <out-dir>/<song>.{pred,ref,input}.txt <song>.pred.{tex,ascii} manifest.json

It can also run a --probe over N random validation windows reporting
exact-match, prefix accuracy, EOS-stop rate and loop rate for
greedy vs beam — to tell "broken model" apart from "broken decoder".

Run:
  python scripts/export_samples.py --checkpoint models/<run>/final \\
      --out-dir models/<run>/samples/step-00016116-redecode
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from train_model import select_sample_songs
from open_fret.render import TabRenderer


def load_model(checkpoint, fp16=True):
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
        if fp16:
            model = model.half()
    model.eval()
    return tok, model


@torch.no_grad()
def decode(tok, model, midi_tokens, max_new_tokens=512, beams=1,
           no_repeat_ngram=0, repetition_penalty=1.0, mode="anchored"):
    from open_fret.generation import generate_tab
    result = generate_tab(model, tok, midi_tokens, max_new_tokens=max_new_tokens,
                          beams=beams, no_repeat_ngram=no_repeat_ngram,
                          repetition_penalty=repetition_penalty, mode=mode)
    return result.tokens, result.stopped_eos


def is_loop(toks, max_unique=3):
    return len(set(toks)) <= max_unique


def write_song(out_dir, renderer, name, midi_tokens, ref_tokens, pred_tokens,
               note=""):
    base = "".join(
        c if (c.isalnum() or c in "-_.") else "_" for c in name)[:120]
    with open(os.path.join(out_dir, base + ".pred.txt"), "w",
              encoding="utf-8") as f:
        f.write(" ".join(pred_tokens))
    with open(os.path.join(out_dir, base + ".ref.txt"), "w",
              encoding="utf-8") as f:
        f.write(" ".join(ref_tokens))
    with open(os.path.join(out_dir, base + ".input.txt"), "w",
              encoding="utf-8") as f:
        f.write(" ".join(midi_tokens))
    with open(os.path.join(out_dir, base + ".pred.tex"), "w",
              encoding="utf-8") as f:
        f.write(renderer.to_alphatex(pred_tokens, title=base))
    with open(os.path.join(out_dir, base + ".pred.ascii"), "w",
              encoding="utf-8") as f:
        f.write(renderer.to_ascii(pred_tokens))
    return base


def export_from_model(tok, model, songs, out_dir, *, title_prefix="",
                      beams=4, decoding_mode="anchored", no_repeat_ngram=0,
                      repetition_penalty=1.0, build_viewer=True):
    """Decode fixed songs with a trained model and write playable artifacts.

    Writes ``<song>.{pred,ref,input}.txt``, ``<song>.pred.{tex,ascii}`` and
    ``manifest.json`` (plus ``viewer.html`` when ``build_viewer`` and the
    repo dist bundle is available), so any run — full training or a single
    HPO trial — leaves generated examples connected to its config. Returns
    the manifest dict. Never raises for viewer failures.
    """
    from open_fret.viewer import build_view_bundle, renderer_for_input, to_html
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"songs": [], "decoding": {"beams": beams, "mode": decoding_mode,
                "no_repeat_ngram": no_repeat_ngram,
                "repetition_penalty": repetition_penalty}}
    sections = []
    dist_js = None
    if build_viewer:
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(root, "ui", "tab_data_viewer", "dist",
                                   "tab-data-viewer.umd.js"), encoding="utf-8") as f:
                dist_js = f.read()
        except OSError:
            dist_js = None
    for s in songs:
        pred, stopped = decode(tok, model, s["midi_tokens"],
                               beams=beams, no_repeat_ngram=no_repeat_ngram,
                               repetition_penalty=repetition_penalty,
                               mode=decoding_mode)
        renderer = renderer_for_input(s["midi_tokens"], s.get("tempo", 120))
        base = write_song(out_dir, renderer, s["name"],
                          s["midi_tokens"], s["tab_tokens"], pred)
        manifest["songs"].append(
            {"name": s["name"], "file": base, "n_pred": len(pred),
             "n_ref": len(s["tab_tokens"]), "stopped_eos": stopped,
             "tempo": s.get("tempo", 120), "loop": is_loop(pred)})
        if dist_js is not None:
            try:
                bundle = build_view_bundle(
                    midi_tokens=s["midi_tokens"], tab_tokens=pred,
                    reference_tab_tokens=s["tab_tokens"],
                    title=f"{title_prefix}{s['name']}" if title_prefix else s["name"],
                    tempo=s.get("tempo", 120))
                sections.append(f"<h2>{base}</h2>\n"
                                + to_html(bundle, dist_js, with_deps=not sections))
            except Exception as e:  # noqa: BLE001 - viewer never blocks export
                print(f"viewer build failed for {s['name']}: {e}")
    if dist_js is not None and sections:
        deps_note = ("<p>Playable predictions: MIDI input (left) vs tab (right). "
                     "First section carries the viewer scripts.</p>\n")
        with open(os.path.join(out_dir, "viewer.html"), "w", encoding="utf-8") as f:
            f.write('<!doctype html><html><head><meta charset="utf-8">'
                    '<title>Sample viewer</title></head><body>\n'
                    + deps_note + "\n<hr>\n".join(sections) + "\n</body></html>\n")
        manifest["viewer"] = "viewer.html"
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-files",
                   default="dataset/processed/val_files.txt")
    p.add_argument("--data-dir", default="dataset/processed/individual")
    p.add_argument("--sample-songs", type=int, default=5)
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--decoding-mode", choices=["anchored", "legacy"], default="anchored")
    p.add_argument("--no-repeat-ngram", type=int, default=3)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--neighbor-search", action="store_true",
                   help="Snap predicted frets to input pitches "
                        "(open_fret.postprocess).")
    p.add_argument("--probe-n", type=int, default=100,
                   help="Random val windows for greedy-vs-beam metrics "
                        "(0 disables).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.neighbor_search and args.decoding_mode == "anchored":
        p.error("Neighbor search would invalidate anchored guarantees; use legacy comparison mode")

    tok, model = load_model(args.checkpoint)
    correct = None
    if args.neighbor_search:
        from open_fret.postprocess import neighbor_search as correct
    os.makedirs(args.out_dir, exist_ok=True)

    songs = select_sample_songs(args.val_files, args.data_dir,
                                k=args.sample_songs)
    manifest = export_from_model(
        tok, model, songs, args.out_dir,
        beams=args.beams, decoding_mode=args.decoding_mode,
        no_repeat_ngram=args.no_repeat_ngram if args.decoding_mode == "legacy" else 0,
        repetition_penalty=args.repetition_penalty)
    manifest["checkpoint"] = args.checkpoint
    manifest["decoding"]["neighbor_search"] = bool(correct)
    if correct is not None:
        # Legacy comparison mode only: snap + rewrite predictions in place.
        for entry in manifest["songs"]:
            base = entry["file"]
            with open(os.path.join(args.out_dir, base + ".pred.txt"),
                      encoding="utf-8") as f:
                pred = f.read().split()
            with open(os.path.join(args.out_dir, base + ".input.txt"),
                      encoding="utf-8") as f:
                midi = f.read().split()
            try:
                fixed = correct(pred, midi)
            except Exception as e:  # noqa: BLE001
                print(f"neighbor_search failed for {entry['name']}: {e}")
                continue
            with open(os.path.join(args.out_dir, base + ".pred.txt"),
                      "w", encoding="utf-8") as f:
                f.write(" ".join(fixed))
    with open(os.path.join(args.out_dir, "manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    for entry in manifest["songs"]:
        print(f"{entry['name'][:50]:50s} n_pred={entry['n_pred']:4d}")

    if args.probe_n > 0:
        from datasets import load_from_disk
        rng = random.Random(args.seed)
        with open(args.val_files, encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        stats = {"greedy": {"exact": 0, "prefix_acc": [], "eos": 0,
                            "loops": 0, "n": 0},
                 "beam": {"exact": 0, "prefix_acc": [], "eos": 0,
                          "loops": 0, "n": 0}}
        for name in rng.sample(names, min(args.probe_n, len(names))):
            with open(os.path.join(args.data_dir, name),
                      encoding="utf-8") as f:
                dd = json.load(f)
            ref = dd["tab"]
            for key, kw in (("greedy", {"beams": 1}),
                            ("beam", {"beams": args.beams,
                                      "no_repeat_ngram": args.no_repeat_ngram,
                                      "repetition_penalty":
                                      args.repetition_penalty})):
                pred, stopped = decode(tok, model, dd["midi"], mode="legacy", **kw)
                st = stats[key]
                st["n"] += 1
                st["exact"] += (pred == ref)
                m = min(len(pred), len(ref))
                st["prefix_acc"].append(
                    sum(a == b for a, b in zip(pred[:m], ref[:m])) / max(m, 1))
                st["eos"] += stopped
                st["loops"] += is_loop(pred)
        for key, st in stats.items():
            n = max(st["n"], 1)
            manifest[f"probe_{key}"] = {
                "n": st["n"], "exact_match_rate": round(st["exact"] / n, 4),
                "mean_prefix_acc": round(sum(st["prefix_acc"]) / n, 4),
                "eos_stop_rate": round(st["eos"] / n, 4),
                "loop_rate": round(st["loops"] / n, 4)}
            print(f"probe/{key}: {manifest[f'probe_{key}']}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
