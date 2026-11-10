"""Blind A/B judging of generated tablatures with the repo's playable viewer.

Serves pairs of HPO-trial (or run-sample) predictions for the same song side
by side — trial identities hidden until you vote — and records A/B/tie/skip
verdicts in SQLite with per-trial Elo ratings. Results export as JSONL
``{song, input_tokens, chosen, rejected, ...}`` triples for later
preference-model training.

Usage:
    python -m venv /tmp/pj-venv && /tmp/pj-venv/bin/pip install -r requirements.txt
    cd helper_apps/pair_judge && /tmp/pj-venv/bin/python server.py \\
        --trials "models/hpo-play-*/trial_*/samples" --db votes.db
    # frontend needs one build: cd web && npm install && npm run build

The viewer itself (AlphaTab + piano roll, vendored in the repo) is mounted
by the React frontend; no model or GPU is needed here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sqlite3
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pairing import ELO_BASE, build_pairs, compute_elo, discover_trials

HERE = os.path.dirname(os.path.abspath(__file__))


def _ensure_repo_on_path(start: str = HERE) -> str | None:
    cur = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(cur, "open_fret", "viewer.py")):
            if cur not in sys.path:
                sys.path.insert(0, cur)
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


REPO_ROOT = _ensure_repo_on_path()


def find_repo_root(start: str = HERE) -> str | None:
    return REPO_ROOT or _ensure_repo_on_path(start)


class Vote(BaseModel):
    pair_id: str
    winner: str  # "left" | "right" | "tie"
    notes: str = ""
    judge: str = "anon"


class Skip(BaseModel):
    pair_id: str
    judge: str = "anon"


SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (label TEXT PRIMARY KEY, path TEXT);
CREATE TABLE IF NOT EXISTS pairs (
    id TEXT PRIMARY KEY, song TEXT NOT NULL,
    left_trial TEXT NOT NULL, right_trial TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, pair_id TEXT NOT NULL UNIQUE,
    winner TEXT, notes TEXT DEFAULT '', judge TEXT DEFAULT 'anon',
    ts TEXT NOT NULL, skipped INTEGER DEFAULT 0);
"""


def open_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def build_view_bundle(input_tokens, tab_tokens, title, tempo=120):
    from open_fret.viewer import build_view_bundle
    return build_view_bundle(midi_tokens=list(input_tokens),
                             tab_tokens=list(tab_tokens), title=title,
                             tempo=tempo)


def create_app(trial_patterns: list[str], db_path: str, songs_limit: int = 0,
               seed: int = 0, repo_root: str | None = None) -> FastAPI:
    repo_root = repo_root or find_repo_root()
    if repo_root is None:
        raise RuntimeError("Cannot locate repo root (ui/tab_data_viewer/dist "
                           "missing). Pass --repo-root explicitly.")
    dist_js = os.path.join(repo_root, "ui", "tab_data_viewer", "dist",
                           "tab-data-viewer.umd.js")
    if not os.path.isfile(dist_js):
        raise RuntimeError(f"Viewer bundle missing: {dist_js} "
                           "(run `npm run build` in ui/tab_data_viewer).")

    trials = discover_trials(trial_patterns)
    if len(trials) < 2:
        raise RuntimeError(f"Need >= 2 trial sample dirs, found {len(trials)} "
                           f"for {trial_patterns}.")
    if songs_limit > 0:
        # Shared panel: first `songs_limit` songs present in every trial.
        common = [s["name"] for s in trials[0]["songs"]]
        for trial in trials[1:]:
            names = {s["name"] for s in trial["songs"]}
            common = [n for n in common if n in names]
        keep = set(common[:songs_limit])
        for trial in trials:
            trial["songs"] = [s for s in trial["songs"] if s["name"] in keep]
        trials = [t for t in trials if t["songs"]]

    pairs = build_pairs(trials, seed=seed)
    if not pairs:
        raise RuntimeError("No shared songs across trials: no pairs to judge.")
    by_label = {t["label"]: t for t in trials}

    conn = open_db(db_path)
    lock = threading.Lock()
    with lock, conn:
        conn.executemany("INSERT OR IGNORE INTO trials(label, path) VALUES (?,?)",
                         [(t["label"], t["path"]) for t in trials])
        conn.executemany("INSERT OR IGNORE INTO pairs(id, song, left_trial, right_trial) VALUES (?,?,?,?)",
                         [(p["id"], p["song"], p["left"], p["right"]) for p in pairs])

    bundle_cache: dict[tuple[str, str], dict] = {}

    def bundle_for(song_name: str, trial_label: str) -> dict:
        # NOTE: the bundle title must stay trial-free — the frontend must
        # not see which trial a side is until after voting (blind judging).
        key = (song_name, trial_label)
        if key not in bundle_cache:
            trial = by_label[trial_label]
            song = next(s for s in trial["songs"] if s["name"] == song_name)
            bundle_cache[key] = build_view_bundle(
                song["input"], song["pred"], song["name"],
                tempo=song.get("tempo", 120))
        return bundle_cache[key]

    def judged_ids() -> set[str]:
        rows = conn.execute("SELECT pair_id FROM votes").fetchall()
        return {r[0] for r in rows}

    def ordered_votes():
        rows = conn.execute("SELECT pair_id, winner FROM votes WHERE skipped = 0 ORDER BY id").fetchall()
        out = []
        for pair_id, winner in rows:
            row = conn.execute("SELECT left_trial, right_trial FROM pairs WHERE id = ?",
                               (pair_id,)).fetchone()
            if row is None:
                continue
            a, b = row["left_trial"], row["right_trial"]
            score = {"left": 1.0, "right": 0.0, "tie": 0.5}[winner]
            out.append({"a": a, "b": b, "score_a": score})
        return out

    app = FastAPI(title="open-fret pair judge")

    @app.get("/api/health")
    def health():
        return {"ok": True, "trials": len(trials), "pairs": len(pairs)}

    @app.get("/api/next")
    def next_pair():
        done = judged_ids()
        remaining = [p for p in pairs if p["id"] not in done]
        total = len(pairs)
        skipped_here = []
        while remaining:
            pair = remaining.pop(0)
            try:
                left = bundle_for(pair["song"], pair["left"])
                right = bundle_for(pair["song"], pair["right"])
            except Exception as e:  # noqa: BLE001 - unrenderable: log & skip
                with lock, conn:
                    conn.execute("INSERT OR IGNORE INTO votes(pair_id, winner, notes, judge, ts, skipped) "
                                 "VALUES (?, '', ?, 'auto', ?, 1)",
                                 (pair["id"],
                                  f"unrenderable: {type(e).__name__}: {str(e)[:160]}",
                                  dt.datetime.now(dt.timezone.utc).isoformat()))
                skipped_here.append(pair["id"])
                continue
            return {"done": False, "pair_id": pair["id"], "song": pair["song"],
                    "left": {"side": "left", "bundle": left},
                    "right": {"side": "right", "bundle": right},
                    "total": total, "remaining": len(remaining),
                    "skipped": len(skipped_here)}
        return {"done": True, "total": total, "remaining": 0}

    @app.post("/api/vote")
    def vote(v: Vote):
        if v.winner not in ("left", "right", "tie"):
            raise HTTPException(400, "winner must be left, right or tie")
        row = conn.execute("SELECT left_trial, right_trial FROM pairs WHERE id = ?",
                           (v.pair_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "unknown pair_id")
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            with lock, conn:
                conn.execute("INSERT INTO votes(pair_id, winner, notes, judge, ts, skipped) "
                             "VALUES (?,?,?,?,?,0)", (v.pair_id, v.winner, v.notes, v.judge, now))
        except sqlite3.IntegrityError:
            raise HTTPException(409, "pair already judged")
        return {"ok": True, "reveal": {"left": row["left_trial"], "right": row["right_trial"]}}

    @app.post("/api/skip")
    def skip(s: Skip):
        row = conn.execute("SELECT id FROM pairs WHERE id = ?", (s.pair_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "unknown pair_id")
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with lock, conn:
            conn.execute(
                "INSERT OR IGNORE INTO votes "
                "(pair_id, winner, notes, judge, ts, skipped) "
                "VALUES (?, NULL, '', ?, ?, 1)",
                (s.pair_id, s.judge, now))
        return {"ok": True}

    @app.get("/api/results")
    def results():
        trials_sorted = sorted(by_label)
        elo = compute_elo(ordered_votes(), trials_sorted)
        tally = conn.execute("SELECT winner, COUNT(*) FROM votes WHERE skipped = 0 GROUP BY winner").fetchall()
        skipped = conn.execute("SELECT COUNT(*) FROM votes WHERE skipped = 1").fetchone()[0]
        table = []
        for t in trials_sorted:
            row = conn.execute("SELECT winner, COUNT(*) FROM votes v JOIN pairs p ON p.id = v.pair_id "
                               "WHERE v.skipped = 0 AND (p.left_trial = ? OR p.right_trial = ?) GROUP BY winner",
                               (t, t)).fetchall()
            counts = {w: c for w, c in row}
            table.append({"trial": t, "elo": round(elo[t], 1),
                          "wins": 0, "decided": sum(counts.values())})
        # wins need side-awareness: count wins where this trial was the voted side
        wins: dict[str, int] = {t: 0 for t in trials_sorted}
        for vrow in conn.execute("SELECT p.left_trial, p.right_trial, v.winner FROM votes v "
                                "JOIN pairs p ON p.id = v.pair_id WHERE v.skipped = 0"):
            a, b, w = vrow["left_trial"], vrow["right_trial"], vrow["winner"]
            if w == "left":
                wins[a] = wins.get(a, 0) + 1
            elif w == "right":
                wins[b] = wins.get(b, 0) + 1
        for entry in table:
            entry["wins"] = wins.get(entry["trial"], 0)
        total = conn.execute("SELECT COUNT(*) FROM votes WHERE skipped = 0").fetchone()[0]
        return {"trials": table, "votes": total, "skipped": skipped,
                "pairs": len(pairs),
                "tally": {w: c for w, c in tally}}

    @app.get("/api/export")
    def export():
        lines = []
        for vrow in conn.execute("SELECT pair_id, winner, notes, judge, ts FROM votes "
                                "WHERE skipped = 0 ORDER BY id"):
            prow = conn.execute("SELECT song, left_trial, right_trial FROM pairs WHERE id = ?",
                                (vrow["pair_id"],)).fetchone()
            song = prow["song"]
            sides = {}
            for side, label in (("left", prow["left_trial"]), ("right", prow["right_trial"])):
                trial = by_label[label]
                song_data = next(s for s in trial["songs"] if s["name"] == song)
                sides[side] = {"trial": label, "tab_tokens": song_data["pred"]}
            inp = next(s for s in by_label[prow["left_trial"]]["songs"] if s["name"] == song)["input"]
            base = {"song": song, "input_tokens": inp,
                    "judge": vrow["judge"], "notes": vrow["notes"],
                    "ts": vrow["ts"], "pair_id": vrow["pair_id"]}
            if vrow["winner"] == "tie":
                base.update({"tie": True, "a": sides["left"], "b": sides["right"]})
            else:
                win, lose = (("left", "right") if vrow["winner"] == "left"
                             else ("right", "left"))
                base.update({"chosen": sides[win], "rejected": sides[lose]})
            lines.append(json.dumps(base))
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""),
                                 media_type="text/plain")

    # Static frontend (Vite build) + locally served viewer runtime, so the
    # judging desk works fully offline: AlphaTab/WaveRoll/viewer UMD, the
    # Bravura fonts (via the documented window.ALPHATAB_FONT override), and
    # the SONiVOX soundfont (fetched as a blob by the frontend, which the
    # viewer picks up through window.__OFV_SOUNDFONT_URL).
    web_dist = os.path.join(HERE, "web", "dist", "index.html")
    vendor_dir = os.path.join(repo_root, "ui", "tab_data_viewer", "vendor")
    dist_dir = os.path.join(repo_root, "ui", "tab_data_viewer", "dist")
    npm_font = os.path.join(repo_root, "tools", "alphatex_parser", "node_modules",
                            "@coderline", "alphatab", "dist", "font")
    if os.path.isfile(web_dist):
        from fastapi.staticfiles import StaticFiles
        app.mount("/assets", StaticFiles(directory=os.path.join(HERE, "web", "dist", "assets")),
                  name="assets")
        app.mount("/vendor/dist", StaticFiles(directory=dist_dir), name="vdist")
        app.mount("/vendor/lib", StaticFiles(directory=vendor_dir), name="vlib")
        if os.path.isdir(npm_font):
            app.mount("/vendor/font", StaticFiles(directory=npm_font), name="vfont")

        @app.get("/", response_class=HTMLResponse)
        def index():
            with open(web_dist, encoding="utf-8") as f:
                return f.read()
    else:
        @app.get("/", response_class=HTMLResponse)
        def index_missing():
            return ("<h1>pair judge API running</h1><p>Build the frontend first: "
                    "<code>cd web && npm install && npm run build</code>.</p>")
    return app


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", action="append", required=True,
                   help="Glob(s) of trial sample dirs (repeatable).")
    p.add_argument("--db", default="pair_judge.db", help="SQLite votes database.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--songs", type=int, default=0, help="Limit songs per trial (0 = all shared).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repo-root", default=None)
    args = p.parse_args(argv)
    import uvicorn
    app = create_app(args.trials, args.db, songs_limit=args.songs,
                     seed=args.seed, repo_root=args.repo_root)
    print(f"Serving pair judge on http://{args.host}:{args.port} "
          f"({'frontend built' if os.path.isfile(os.path.join(HERE, 'web', 'dist', 'index.html')) else 'NO FRONTEND BUILD — API only'})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
