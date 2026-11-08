"""Read-only validation diagnostics; never changes training or scores the test set."""

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_fret.transcription_diagnostics import (compare_streams, conditioning, event_stream,
                                                family_key, group_onsets, repetition_conflicts, structure)
from open_fret.splits import group_id_from_filename


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def event_table(run, step, out):
    directory = run / "samples" / f"step-{step:08d}"
    manifest = json.loads((directory / "manifest.json").read_text())
    sections = []
    for song in manifest["songs"]:
        if "error" in song:
            continue
        base = directory / song["file"]
        midi = Path(str(base) + ".input.txt").read_text().split()
        pred = Path(str(base) + ".pred.txt").read_text().split()
        tuning, capo = conditioning(midi)
        source = group_onsets(event_stream(midi, "midi")[0])
        generated = group_onsets(event_stream(pred, "tab", tuning, capo)[0])
        rows = []
        for tick in sorted(source.keys() | generated.keys()):
            expected, actual = source.get(tick, []), generated.get(tick, [])
            duplicate = len(actual) > len({e["string"] for e in actual})
            css = ' class="bad"' if len(actual) > len(expected) or duplicate else ""
            pitches = lambda events: ", ".join(str(e["pitch"]) for e in events) or "—"
            assignments = ", ".join(f'S{e["string"]}' for e in actual)
            rows.append(f"<tr{css}><td>{tick}</td><td>{pitches(expected)}</td>"
                        f"<td>{pitches(actual)}</td><td>{assignments}</td>"
                        f"<td>{'same-string conflict' if duplicate else ''}</td></tr>")
        sections.append(f"<h2>{html.escape(song['name'])}</h2>"
                        "<table><tr><th>Tick</th><th>Input attacks (MIDI pitch)</th>"
                        "<th>Predicted attacks</th><th>Predicted strings</th><th>Conflict</th></tr>"
                        + "\n".join(rows) + "</table>")
    (out / "event_comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Raw attack comparison</title>'
        '<style>body{font-family:system-ui;max-width:1200px;margin:auto}table{border-collapse:collapse}'
        'td,th{padding:4px 12px;border:1px solid #ddd}.bad{background:#ffe0e0}</style>'
        f'<h1>Raw input/prediction attacks — checkpoint {step}</h1>'
        '<p>480 ticks per quarter. Red: extra attacks or duplicate strings at that exact tick. '
        'Sustaining notes are not new attacks. No post-processing or AlphaTex rendering is involved.</p>'
        + "\n".join(sections))


def render_audit(run, step, out):
    from open_fret.viewer import renderer_for_input
    from open_fret.render import TabRenderer, _duration_token
    directory = run / "samples" / f"step-{step:08d}"
    manifest = json.loads((directory / "manifest.json").read_text())
    requests, metadata = [], {}
    for song in manifest["songs"]:
        if "error" in song:
            continue
        base = directory / song["file"]
        midi = Path(str(base) + ".input.txt").read_text().split()
        tuning, capo = conditioning(midi)
        for kind, suffix in [("prediction", ".pred.txt"), ("reference", ".ref.txt")]:
            tokens = Path(str(base) + suffix).read_text().split()
            tex = (Path(str(base) + ".pred.tex").read_text() if kind == "prediction" else
                   renderer_for_input(midi, song["tempo"]).to_alphatex(tokens))
            name = song["name"] + " " + kind
            notes = event_stream(tokens, "tab", tuning, capo)[0]
            _, gaps = TabRenderer._collect_chords(tokens)
            metadata[name] = {"kind": kind, "raw_notes": len(notes),
                              "raw_onsets": len(group_onsets(notes)),
                              "first_dotted_directive_group_0based": next((i for i, gap in enumerate(gaps)
                                  if (_duration_token(gap) or ":4").endswith(".")), None)}
            requests.append({"name": name, "tex": tex})
    header = '\\title "probe"\n\\tempo 120\n.\n\\track "Guitar"\n\\tuning E4 B3 G3 D3 A2 E2\n'
    requests.extend([{"name": "bad_dotted_prefix", "tex": header + ":8. 0.1 0.2"},
                     {"name": "correct_dotted_180_ticks", "tex": header + ":16 0.1{d} 0.2{d}"}])
    result = subprocess.run(["node", str(ROOT / "tools/alphatex_parser/audit_render.cjs")],
                            input="\n".join(json.dumps(r) for r in requests) + "\n",
                            capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        row = json.loads(line)
        row.update(metadata.get(row["name"], {}))
        row["imported_notes"] = sum(len(g["pitches"]) for g in row.get("groups", []))
        row["imported_onsets"] = len(row.get("groups", []))
        rows.append(row)
    write(out / "render_audit.json", rows)
    print(f"Wrote {len(rows)} independent import comparisons to {out}")


def split_audit(data, n):
    rng = random.Random(42)
    split_groups, families, profiles, examples = {}, defaultdict(lambda: defaultdict(set)), {}, []
    introductory = {"train": defaultdict(list), "validation": defaultdict(list)}
    for split, filename in [("train", "train_files.txt"), ("validation", "val_files.txt"), ("test", "test_files.txt")]:
        groups, c0, caps = set(), defaultdict(list), Counter()
        for name in (data / filename).read_text().splitlines():
            group = group_id_from_filename(name)
            groups.add(group)
            capo = int(name.rsplit("__c", 1)[1].split("__", 1)[0])
            caps[capo] += 1
            if capo == 0 and split != "test":
                c0[group].append(name)
                if name.endswith("__w0.json"):
                    introductory[split][group].append(name)
        split_groups[split] = groups
        for group in sorted(groups):
            families[family_key(group)][split].add(group)
        # Test filenames/IDs ONLY for leakage audit; no test music/model evaluation.
        profile = {"source_groups": len(groups), "window_counts_by_capo": dict(caps)}
        if split != "test":
            rows = []
            for group in rng.sample(sorted(c0), min(n, len(c0))):
                name = rng.choice(c0[group])
                record = json.loads((data / "individual" / name).read_text())
                notes, _ = event_stream(record["midi"], "midi")
                stats = structure(notes)
                stats.update(file=name, reference_repeated_trigrams=repetition_conflicts(record["tab"]),
                             strings=len(record["tuning"]), tuning=record["tuning"],
                             starts_with_note=record["tab"][0].startswith("TAB_"),
                             end_tick=record["end_tick"], start_tick=record["start_tick"])
                rows.append(stats)
            profile.update(sampled_source_balanced_capo_zero=rows,
                           pure_monophonic_windows=sum(r["chord_onsets"] == 0 for r in rows),
                           almost_monophonic_windows=sum(r["chord_rate"] <= 0.05 for r in rows),
                           note_onsets=sum(r["onsets"] for r in rows),
                           chord_onsets=sum(r["chord_onsets"] for r in rows))
        profiles[split] = profile
    train_val = [v for v in families.values() if "train" in v and "validation" in v]
    crossing = [v for v in families.values() if len(v) > 1]
    for value in crossing[:25]:
        examples.append({s: sorted(g) for s, g in value.items()})
    train_artists = {g.partition("__")[0] for g in split_groups["train"]}
    exact_examples, exact_families, exact_val_windows, exact_io_windows = [], 0, 0, 0
    for candidate in train_val:
        hashes = {}
        found_family = False
        for split in ("train", "validation"):
            for group in sorted(candidate[split]):
                for filename in introductory[split][group]:
                    record = json.loads((data / "individual" / filename).read_text())
                    tick, events, notes = 0, [], 0
                    for token in record["midi"]:
                        if token.startswith("TS_"):
                            tick += int(token[3:])
                        elif token.startswith(("NO_", "NF_")):
                            events.append((tick, token))
                            notes += token.startswith("NO_")
                    if notes < 16:
                        continue
                    digest = hashlib.sha256(json.dumps([events, tick]).encode()).hexdigest()
                    if split == "train":
                        hashes.setdefault(digest, []).append((filename, record["tab"], record["midi"]))
                    elif digest in hashes:
                        exact_val_windows += 1
                        found_family = True
                        identical = [name for name, tab, midi in hashes[digest]
                                     if tab == record["tab"] and midi == record["midi"]]
                        exact_io_windows += bool(identical)
                        if len(exact_examples) < 20:
                            exact_examples.append({"train": identical[0] if identical else hashes[digest][0][0],
                                                   "validation": filename,
                                                   "identical_input_and_reference_tokens": bool(identical),
                                                   "notes": notes, "ticks": tick})
        exact_families += found_family
    return {"seed": 42, "profile_sample_size_per_train_val": n, "test_music_read": False,
            "source_group_overlap_counts": {
                f"{a}_{b}": len(split_groups[a] & split_groups[b])
                for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]},
            "heuristic_cross_split_version_families": len(crossing),
            "heuristic_train_validation_version_families": len(train_val),
            "validation_groups_with_training_version_family": sum(len(v.get("validation", set()))
                                                                    for v in train_val),
            "version_family_examples": examples,
            "confirmed_exact_train_val_intro_windows": exact_val_windows,
            "confirmed_exact_train_val_io_intro_windows": exact_io_windows,
            "version_families_with_exact_train_val_intro": exact_families,
            "exact_intro_examples": exact_examples,
            "validation_sources_with_training_artist": sum(g.partition("__")[0] in train_artists
                                                             for g in split_groups["validation"]),
            "splits": profiles}


def decode_ablation(run, step, out):
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    torch.set_num_threads(4)
    checkpoint = run / f"checkpoint-{step}"
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"No local checkpoint at {checkpoint}")
    metadata = json.loads((run / "run.json").read_text())
    tok = AutoTokenizer.from_pretrained(metadata["tokenizer"]["dir"], local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).cpu().eval()
    directory = run / "samples" / f"step-{step:08d}"
    manifest = json.loads((directory / "manifest.json").read_text())
    type_ids = {"TAB": [], "TS": [], "EOS": [tok.eos_token_id]}
    for text, index in tok.get_vocab().items():
        if text.startswith("TAB_"): type_ids["TAB"].append(index)
        elif text.startswith("TS_"): type_ids["TS"].append(index)
    results, teacher = [], []
    for song in manifest["songs"]:
        base = directory / song["file"]
        midi = Path(str(base) + ".input.txt").read_text().split()
        ref = Path(str(base) + ".ref.txt").read_text().split()
        train_input = tok(" ".join(midi) + " " + tok.eos_token, return_tensors="pt")
        labels = tok(" ".join(ref) + " " + tok.eos_token, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            logits = model(**train_input, labels=labels).logits[0].float()
            probs = logits.softmax(-1)
            class_stats = {}
            for kind, ids in type_ids.items():
                mask = torch.isin(labels[0], torch.tensor(ids))
                if mask.any():
                    class_stats[kind] = {"positions": int(mask.sum()),
                        "mean_ce": float(torch.nn.functional.cross_entropy(logits[mask], labels[0][mask])),
                        "mean_correct_type_probability": float(probs[mask][:, ids].sum(-1).mean()),
                        "mean_TAB_probability": float(probs[mask][:, type_ids["TAB"]].sum(-1).mean())}
            teacher.append({"song": song["name"], "classes": class_stats})
        for with_eos in [False, True]:
            enc = tok(" ".join(midi) + (" " + tok.eos_token if with_eos else ""), return_tensors="pt")
            assert enc["input_ids"].shape[1] <= 512
            for ngram in [3, 0]:
                start = time.monotonic()
                with torch.inference_mode():
                    ids = model.generate(**enc, num_beams=4, do_sample=False,
                                         max_new_tokens=512, no_repeat_ngram_size=ngram,
                                         early_stopping=True)[0].tolist()
                pred = tok.decode(ids, skip_special_tokens=True).split()
                name = f"{song['file']}.eos-{int(with_eos)}.ngram-{ngram}.txt"
                (out / name).write_text(" ".join(pred))
                result = {"song": song["name"], "input_eos": with_eos, "no_repeat_ngram_size": ngram,
                          "ended_with_eos": ids[-1] == tok.eos_token_id,
                          "generated_tokens_including_eos": len(ids) - 1,
                          "hit_cap_without_eos": len(ids) - 1 == 512 and ids[-1] != tok.eos_token_id,
                          "seconds": time.monotonic() - start,
                          **compare_streams(midi, ref, pred)}
                results.append(result)
                write(out / "decode_ablation.json", {"checkpoint": str(checkpoint),
                      "device": "CPU float32", "beams": 4, "max_new_tokens": 512,
                      "teacher_forced": teacher, "results": results})
                print(song["file"], "eos", with_eos, "ngram", ngram,
                      "notes", result["prediction"]["notes"], "chords", result["prediction"]["chord_onsets"],
                      "f1", round(result["timed_pitch"]["f1"], 3), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile-sources", type=int, default=512)
    parser.add_argument("--decode-ablation", action="store_true")
    parser.add_argument("--render-audit", action="store_true")
    args = parser.parse_args()
    run, out = Path(args.run_dir).resolve(), Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.decode_ablation:
        decode_ablation(run, args.checkpoint_step, out)
        return
    if args.render_audit:
        render_audit(run, args.checkpoint_step, out)
        return
    rows = []
    for directory in sorted((run / "samples").glob("step-*")):
        if not directory.name[5:].isdigit(): continue
        manifest = json.loads((directory / "manifest.json").read_text())
        for song in manifest["songs"]:
            if "error" in song: continue
            base = directory / song["file"]
            streams = [Path(str(base) + suffix).read_text().split()
                       for suffix in (".input.txt", ".ref.txt", ".pred.txt")]
            rows.append({"step": manifest["step"], "song": song["name"], "file": song["file"],
                         "eval_loss": manifest["eval_metrics"].get("eval_loss"),
                         **compare_streams(*streams)})
    write(out / "saved_samples.json", {"checkpoint_step": args.checkpoint_step, "rows": rows})
    meta = json.loads((run / "run.json").read_text())
    data = Path(meta["data"]["cache_dir"]).parent
    write(out / "split_audit.json", split_audit(data, args.profile_sources))
    event_table(run, args.checkpoint_step, out)
    print(f"Wrote {len(rows)} saved-sample comparisons and split/profile audit to {out}")


if __name__ == "__main__":
    main()
