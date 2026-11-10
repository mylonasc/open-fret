"""export_from_model writes playable artifacts for any run or HPO trial."""
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_loss_cli import fixture_cache  # noqa: E402
from train_model import select_sample_songs  # noqa: E402
from export_samples import export_from_model  # noqa: E402


def _tiny_model(vocab_size):
    return T5ForConditionalGeneration(T5Config(
        vocab_size=vocab_size, d_model=32, d_ff=64, num_layers=1,
        num_decoder_layers=1, num_heads=2, pad_token_id=0,
        eos_token_id=3, decoder_start_token_id=0))


def test_export_from_model_writes_connected_artifacts(fixture_cache, tmp_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(fixture_cache / "tokenizer"))
    model = _tiny_model(len(tok))
    model.eval()
    songs = select_sample_songs(str(fixture_cache / "fixture_files.txt"),
                                str(fixture_cache / "individual"), k=2)
    assert len(songs) == 2
    out = tmp_path / "trial_export"
    manifest = export_from_model(tok, model, songs, str(out), beams=1,
                                 title_prefix="trial 0 — ")
    assert [s["name"] for s in manifest["songs"]] == [s["name"] for s in songs]
    for entry in manifest["songs"]:
        base = entry["file"]
        for suffix in (".pred.txt", ".ref.txt", ".input.txt",
                       ".pred.tex", ".pred.ascii"):
            assert (out / f"{base}{suffix}").exists()
        assert entry["n_pred"] > 0
    assert (out / "manifest.json").exists()
    assert (out / "viewer.html").read_text().count("ofv-") > 0
    assert manifest["decoding"]["beams"] == 1
    assert (out / "viewer.html").read_text().count("trial 0") > 0
    assert manifest["songs"][0]["tempo"] == 120


def test_export_from_model_skips_viewer_without_dist(fixture_cache, tmp_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(fixture_cache / "tokenizer"))
    model = _tiny_model(len(tok))
    model.eval()
    songs = select_sample_songs(str(fixture_cache / "fixture_files.txt"),
                                str(fixture_cache / "individual"), k=1)
    out = tmp_path / "no_viewer"
    manifest = export_from_model(tok, model, songs, str(out), beams=1,
                                 build_viewer=False)
    assert len(manifest["songs"]) == 1
    assert "viewer" not in manifest
    assert not (out / "viewer.html").exists()
    assert (out / (manifest["songs"][0]["file"] + ".pred.txt")).exists()
