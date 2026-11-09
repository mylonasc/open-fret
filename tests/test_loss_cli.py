"""Fixture-only CLI integration. No downloads, GPU, or production data."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from open_fret.alphatex import AlphaTexParser, PARSER_VERSION
from open_fret.sequences import AlignedWindower, SequenceBuilder


def load_script(filename):
    spec = importlib.util.spec_from_file_location("fixture_" + filename, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fixture_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("loss_cli")
    data = root / "individual"
    data.mkdir()
    track = AlphaTexParser().parse_text(
        "\\track\n\\tuning E5 B4 G4 D4 A3 E3\n:4 0.1 5.2 9.3 :8 0.6 0.6")[0]
    tokens = set()
    names = []
    for capo in (0, 2):
        steps, tuning = SequenceBuilder().build_steps(track, capo=capo)
        window = AlignedWindower(64, 64).window(steps, tuning, capo=capo)[0]
        name = f"song__c{capo}.json"
        (data / name).write_text(json.dumps({"midi": window.midi_tokens,
                                            "tab": window.tab_tokens,
                                            "source": f"synthetic-{capo}.tex",
                                            "parser_version": PARSER_VERSION}))
        tokens.update(window.midi_tokens + window.tab_tokens)
        names.append(name)
    (root / "vocab.json").write_text(json.dumps(sorted(tokens)))
    split = root / "fixture_files.txt"
    split.write_text("\n".join(names))
    load_script("02_tokenize.py").build_tokenizer(str(root / "vocab.json"), str(root / "tokenizer"))
    load_script("04_cache_dataset.py").create_cache(
        tokenizer_dir=str(root / "tokenizer"), data_dir=str(data),
        train_list=str(split), val_list=str(split), test_list=str(root / "no-test.txt"),
        cache_dir=str(root / "cache"), max_length=64)
    # Deliberately same tiny synthetic rows in both splits: plumbing only,
    # not a quality estimate or a development/test benchmark.
    return root


@pytest.mark.parametrize("flag,value", [(None, None), ("--pitch-loss-alpha", "0.3"),
                                       ("--equivalence-smoothing", "0.1"),
                                       ('--sampling-strategy', 'source')])
def test_training_cli_from_fixture_cache(fixture_cache, flag, value):
    out = fixture_cache / (flag or "ce").removeprefix("--")
    command = [sys.executable, str(ROOT / "scripts/train_model.py"),
               "--tokenizer-dir", str(fixture_cache / "tokenizer"),
               "--cache-dir", str(fixture_cache / "cache"),
               "--output-dir", str(out), "--max-steps", "1", "--batch-size", "1",
               "--grad-accum", "1", "--workers", "0", "--sample-songs", "0",
               "--save-steps", "1", "--eval-steps", "1", "--warmup-steps", "0"]
    if flag:
        command.extend([flag, value])
    completed = subprocess.run(command, cwd=ROOT,
                               env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                    "CUDA_VISIBLE_DEVICES": ""},
                               capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((out / "run.json").read_text())
    saved = json.loads((out / "final/config.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["result"]["global_steps"] == 1
    assert manifest["loss"] == saved["open_fret_loss"]
    assert manifest["token_metadata"] == saved["open_fret_token_metadata"]
    assert (out / "final/tokenizer.json").exists()
    if flag == '--sampling-strategy':
        assert saved['open_fret_sampling_mode'] == 'source'
        refused = subprocess.run(command[:], cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                 capture_output=True, text=True, timeout=90)
        assert refused.returncode != 0  # existing fresh-run directory is protected
        without_policy = [v for v in command[:-2]] + ['--resume', '--max-steps', '2']
        refused = subprocess.run(without_policy, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                 capture_output=True, text=True, timeout=90)
        assert refused.returncode != 0 and 'sampling/source identity differs' in refused.stderr
        resumed = subprocess.run(command + ['--resume', '--max-steps', '2'], cwd=ROOT,
                                  env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'},
                                  capture_output=True, text=True, timeout=90)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr


def test_cpu_fresh_resume_and_reload(fixture_cache):
    out = fixture_cache / "resume-integration"
    command = [sys.executable, str(ROOT / "scripts/train_model.py"),
               "--tokenizer-dir", str(fixture_cache / "tokenizer"),
               "--cache-dir", str(fixture_cache / "cache"),
               "--output-dir", str(out), "--batch-size", "1", "--grad-accum", "1",
               "--workers", "0", "--sample-songs", "0", "--save-steps", "1",
               "--eval-steps", "1", "--warmup-steps", "0", "--decay-start-step", "10"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": ""}

    def run(*extra):
        return subprocess.run(command + list(extra), cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=90)

    # Explicit resume must not initialize a fresh run or create output files.
    missing = run("--resume", "--max-steps", "1")
    assert missing.returncode != 0 and "No checkpoint found" in missing.stderr
    assert not out.exists()
    fresh = run("--max-steps", "1")
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert "Starting fresh training run" in fresh.stdout
    state1 = json.loads((out / "checkpoint-1/trainer_state.json").read_text())
    assert state1["global_step"] == 1
    assert (out / "checkpoint-1/rng_state.pth").exists()
    # Nonempty output must not be overwritten by a second fresh run.
    refused = run("--max-steps", "2")
    assert refused.returncode != 0 and "not empty" in refused.stderr
    resumed = run("--resume", "--max-steps", "2")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Resuming training from checkpoint" in resumed.stdout
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["mode"] == "RESUMING"
    assert manifest["result"]["global_steps"] == 2
    # Adafactor step counters would be 1 if optimizer state had been reset.
    optimizer = torch.load(out / "checkpoint-2/optimizer.pt", weights_only=True)
    assert optimizer["state"]
    assert {s["step"] for s in optimizer["state"].values()} == {2}
    scheduler = torch.load(out / "checkpoint-2/scheduler.pt", weights_only=True)
    assert scheduler["last_epoch"] == 2
    model = T5ForConditionalGeneration.from_pretrained(out / "final").eval()
    tok = AutoTokenizer.from_pretrained(out / "final")
    inputs = tok("CAPO_0", return_tensors="pt")
    with torch.no_grad():
        assert model.generate(**inputs, max_new_tokens=2).shape[0] == 1


def test_cached_uncached_are_identical_and_never_truncated(fixture_cache):
    from datasets import load_from_disk
    from scripts.dataloader import TabDataset
    tok = AutoTokenizer.from_pretrained(fixture_cache / 'tokenizer')
    rows = TabDataset(str(fixture_cache / 'fixture_files.txt'), str(fixture_cache / 'individual'), tok, 64)
    cached = load_from_disk(str(fixture_cache / 'cache'))['train']
    for index in range(len(rows)):
        assert {k: v.tolist() for k, v in rows[index].items()} == cached[index]
    tiny = TabDataset(str(fixture_cache / 'fixture_files.txt'), str(fixture_cache / 'individual'), tok, 3)
    with pytest.raises(RuntimeError, match='never truncate'):
        tiny[0]
