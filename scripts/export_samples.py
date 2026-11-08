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
           no_repeat_ngram=0, repetition_penalty=1.0):
    enc = tok(" ".join(midi_tokens), return_tensors="pt",
              truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=max(beams, 1),
        no_repeat_ngram_size=no_repeat_ngram or None,
        repetition_penalty=(repetition_penalty or None),
        early_stopping=True)
    ids = out[0].tolist()
    stopped_eos = (ids and ids[-1] == tok.eos_token_id)
    toks = tok.decode(out[0], skip_special_tokens=True).split()
    return toks, stopped_eos


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-files",
                   default="dataset/processed/val_files.txt")
    p.add_argument("--data-dir", default="dataset/processed/individual")
    p.add_argument("--sample-songs", type=int, default=5)
    p.add_argument("--beams", type=int, default=4)
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

    tok, model = load_model(args.checkpoint)
    renderer = TabRenderer()
    correct = None
    if args.neighbor_search:
        from open_fret.postprocess import neighbor_search as correct
    os.makedirs(args.out_dir, exist_ok=True)

    songs = select_sample_songs(args.val_files, args.data_dir,
                                k=args.sample_songs)
    manifest = {"checkpoint": args.checkpoint,
                "decoding": {"beams": args.beams,
                             "no_repeat_ngram": args.no_repeat_ngram,
                             "repetition_penalty": args.repetition_penalty,
                             "neighbor_search": bool(correct)},
                "songs": []}
    for s in songs:
        pred, stopped = decode(tok, model, s["midi_tokens"],
                               beams=args.beams,
                               no_repeat_ngram=args.no_repeat_ngram,
                               repetition_penalty=args.repetition_penalty)
        if correct is not None:
            try:
                pred = correct(pred, s["midi_tokens"])
            except Exception as e:  # noqa: BLE001
                print(f"neighbor_search failed for {s['name']}: {e}")
        base = write_song(args.out_dir, renderer, s["name"],
                          s["midi_tokens"], s["tab_tokens"], pred)
        manifest["songs"].append(
            {"name": s["name"], "file": base, "n_pred": len(pred),
             "n_ref": len(s["tab_tokens"]), "stopped_eos": stopped,
             "loop": is_loop(pred)})
        print(f"{s['name'][:50]:50s} n_pred={len(pred):4d} "
              f"eos={stopped!s:5s} loop={is_loop(pred)!s:5s}")

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
                pred, stopped = decode(tok, model, dd["midi"], **kw)
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
