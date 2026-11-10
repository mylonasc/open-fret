"""End-to-end API tests: discover trials, serve blind pairs, vote, rank, export."""
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server
from pairing import build_pairs, compute_elo, discover_trials


MIDI = "CAPO_0 TUNING_64_59_55_50_45_40 NO_64 TS_120 NF_64".split()
TAB_A = "TAB_1_0 TS_120".split()
TAB_B = "TAB_1_5 TS_120".split()


def make_trial(root, name, tab):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    songs = [{"name": "song_one", "file": "song_one"},
             {"name": "song_two", "file": "song_two"}]
    with open(os.path.join(d, "samples", "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"songs": songs}, f)
    for song in songs:
        for ext, toks in (("input", MIDI), ("pred", tab),
                          ("ref", TAB_A)):
            with open(os.path.join(d, "samples", f"{song['file']}.{ext}.txt"),
                      "w", encoding="utf-8") as f:
                f.write(" ".join(toks))
    return os.path.join(d, "samples")


def make_client(tmp_path, monkeypatch):
    a = make_trial(str(tmp_path), "trial_a", TAB_A)
    b = make_trial(str(tmp_path), "trial_b", TAB_B)
    monkeypatch.chdir(tmp_path)
    app = server.create_app([a, b], str(tmp_path / "votes.db"))
    return TestClient(app), a, b


def test_discover_and_blind_pairs(tmp_path):
    a = make_trial(str(tmp_path), "trial_a", TAB_A)
    b = make_trial(str(tmp_path), "trial_b", TAB_B)
    trials = discover_trials([a, b])
    assert [t["label"].split("/")[-1] for t in trials] == ["trial_a", "trial_b"]
    assert len(trials[0]["songs"]) == 2
    pairs = build_pairs(trials, seed=1)
    # 2 songs x 1 unordered pair each
    assert len(pairs) == 2
    assert {p["id"] for p in pairs} == {"p0001", "p0002"}
    for p in pairs:
        assert {p["left"].split("/")[-1], p["right"].split("/")[-1]} == {"trial_a", "trial_b"}
        assert p["left"] != p["right"]


def test_elo_updates():
    elo = compute_elo([{"a": "x", "b": "y", "score_a": 1.0}], ["x", "y"])
    assert elo["x"] > 1500 > elo["y"]
    elo = compute_elo([{"a": "x", "b": "y", "score_a": 0.5}], ["x", "y"])
    assert elo["x"] == elo["y"] == 1500.0


def test_vote_flow_and_export(tmp_path, monkeypatch):
    client, a, b = make_client(tmp_path, monkeypatch)
    first = client.get("/api/next").json()
    assert not first["done"] and first["pair_id"].startswith("p")
    assert first["left"]["bundle"]["alpha_tex"] and first["right"]["bundle"]["alpha_tex"]
    # Trial labels must stay hidden pre-vote (bundles carry no labels).
    dumped = json.dumps(first)
    assert "trial_a" not in dumped and "trial_b" not in dumped
    r = client.post("/api/vote", json={"pair_id": first["pair_id"],
                                       "winner": "left", "notes": "cleaner",
                                       "judge": "tester"})
    assert r.status_code == 200
    reveal = r.json()["reveal"]
    assert reveal["left"] != reveal["right"]
    assert {reveal["left"].split("/")[-1], reveal["right"].split("/")[-1]} == {"trial_a", "trial_b"}
    # Same pair cannot be voted twice.
    assert client.post("/api/vote", json={"pair_id": first["pair_id"],
                                          "winner": "right"}).status_code == 409
    second = client.get("/api/next").json()
    assert second["pair_id"] != first["pair_id"] or second.get("done")
    client.post("/api/vote", json={"pair_id": second["pair_id"], "winner": "tie"})
    res = client.get("/api/results").json()
    assert res["votes"] == 2
    assert {t["trial"].split("/")[-1] for t in res["trials"]} == {"trial_a", "trial_b"}
    assert any(t["elo"] != 1500.0 for t in res["trials"])
    exp = client.get("/api/export").text.strip().split("\n")
    assert len(exp) == 2
    first_vote = json.loads(exp[0])
    assert set(first_vote) >= {"song", "input_tokens", "chosen", "rejected",
                               "judge", "pair_id"}
    assert first_vote["judge"] == "tester"
    assert first_vote["chosen"]["tab_tokens"] != first_vote["rejected"]["tab_tokens"]
    tie_lines = [json.loads(line) for line in exp if json.loads(line).get("tie")]
    assert len(tie_lines) == 1 and set(tie_lines[0]) >= {"a", "b"}


def test_skip_advances_queue(tmp_path, monkeypatch):
    client, _, _ = make_client(tmp_path, monkeypatch)
    first = client.get("/api/next").json()
    assert client.post("/api/skip",
                       json={"pair_id": first["pair_id"]}).status_code == 200
    second = client.get("/api/next").json()
    assert second.get("done") or second["pair_id"] != first["pair_id"]
    res = client.get("/api/results").json()
    assert res["skipped"] == 1 and res["votes"] == 0


def test_unrenderable_pair_auto_skips_with_reason(tmp_path, monkeypatch):
    ok_pred = "TAB_1_0 TS_120".split()
    bad_pred = "TAB_1_0 TAB_1_5 TS_120".split()
    d = tmp_path / "fix"
    d.mkdir()
    for trial in ("trial_a", "trial_b"):
        td = d / trial / "samples"
        td.mkdir(parents=True)
        (td / "manifest.json").write_text(json.dumps(
            {"songs": [{"name": "song_ok", "file": "song_ok"},
                       {"name": "song_bad", "file": "song_bad"}]}))
        for name, pred in (("song_ok", ok_pred), ("song_bad", bad_pred)):
            (td / f"{name}.input.txt").write_text(
                "CAPO_0 TUNING_64_59_55_50_45_40 NO_64 TS_120 NF_64")
            (td / f"{name}.pred.txt").write_text(" ".join(pred))
            (td / f"{name}.ref.txt").write_text(" ".join(ok_pred))
    monkeypatch.chdir(d)
    app = server.create_app([str(d / "trial_a" / "samples"),
                             str(d / "trial_b" / "samples")],
                            str(d / "votes.db"))
    client = TestClient(app)
    first = client.get("/api/next").json()
    # song_bad's pred has a same-onset conflict: auto-skipped with a reason,
    # and the renderable pair is served instead.
    assert first.get("done") or first["song"] == "song_ok"
    res = client.get("/api/results").json()
    assert res["skipped"] == 1 and res["votes"] == 0
