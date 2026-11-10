"""Pair construction, blinding, and Elo ratings for tab A/B judging."""
from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path


def trial_label(trial_dir: str) -> str:
    """Human-readable id, e.g. ``hpo-play-pos05/trial_3``.

    Trial sample dirs are conventionally ``.../<study>/<trial>/samples``;
    label with study + trial so trials from different studies never collide.
    """
    p = Path(trial_dir)
    if p.name == "samples" and len(p.parts) >= 3:
        return f"{p.parts[-3]}/{p.parts[-2]}"
    if len(p.parts) >= 2:
        return f"{p.parts[-2]}/{p.parts[-1]}"
    return p.name


def discover_trials(patterns: list[str]) -> list[dict]:
    """Find trial sample dirs (each with manifest.json + {file}.input/.pred.txt)."""
    found: dict[str, dict] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            manifest = os.path.join(path, "manifest.json")
            if not os.path.isdir(path) or not os.path.isfile(manifest):
                continue
            try:
                with open(manifest, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            songs = []
            for song in data.get("songs", []) if isinstance(data, dict) else []:
                base = song.get("file")
                if not base:
                    continue
                need = {s: os.path.join(path, f"{base}.{s}.txt") for s in ("input", "pred")}
                if not all(os.path.isfile(p) for p in need.values()):
                    continue
                try:
                    toks = {k: open(p, encoding="utf-8").read().split()
                            for k, p in need.items()}
                except OSError:
                    continue
                if not toks["input"] or not toks["pred"]:
                    continue
                songs.append({"name": song.get("name", base), "file": base,
                              "input": toks["input"], "pred": toks["pred"],
                              "tempo": song.get("tempo", 120)})
            if songs:
                found[os.path.abspath(path)] = {
                    "path": os.path.abspath(path),
                    "label": trial_label(path),
                    "songs": songs,
                }
    return [found[k] for k in sorted(found)]


def build_pairs(trials: list[dict], seed: int = 0) -> list[dict]:
    """All unordered trial pairs per shared song, shuffled, sides randomized.

    Returns dicts with stable ids ``p0001...`` and a hidden coin flip: the
    judge sees only ``left``/``right`` until voting reveals the mapping.
    """
    rng = random.Random(seed)
    by_song: dict[str, list[int]] = {}
    for i, trial in enumerate(trials):
        for song in trial["songs"]:
            by_song.setdefault(song["name"], []).append(i)
    specs = []
    for song in sorted(by_song):
        idx = by_song[song]
        for a_pos in range(len(idx)):
            for b_pos in range(a_pos + 1, len(idx)):
                specs.append((song, idx[a_pos], idx[b_pos]))
    rng.shuffle(specs)
    pairs = []
    for n, (song, a, b) in enumerate(specs, 1):
        flip = rng.random() < 0.5
        pairs.append({"id": f"p{n:04d}", "song": song,
                      "left": trials[b if flip else a]["label"],
                      "right": trials[a if flip else b]["label"]})
    return pairs


ELO_BASE = 1500.0
ELO_K = 24.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo(votes_in_order: list[dict], trials: list[str]) -> dict[str, float]:
    """Elo per trial from (trial_a, trial_b, score_a) votes in rowid order."""
    ratings = {t: ELO_BASE for t in trials}
    for vote in votes_in_order:
        a, b, score_a = vote["a"], vote["b"], vote["score_a"]
        if a not in ratings or b not in ratings:
            continue
        exp_a = expected_score(ratings[a], ratings[b])
        ratings[a] += ELO_K * (score_a - exp_a)
        ratings[b] += ELO_K * ((1.0 - score_a) - (1.0 - exp_a))
    return ratings
