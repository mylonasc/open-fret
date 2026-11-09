import sys
import json
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.training import resolve_resume_checkpoint


def test_fresh_start_returns_none(tmp_path):
    outdir = tmp_path / "models"
    outdir.mkdir()
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        resolve_resume_checkpoint(str(outdir), True)
    assert resolve_resume_checkpoint(str(outdir), False) is None
    assert resolve_resume_checkpoint(str(outdir), None) is None


def test_resume_finds_latest(tmp_path):
    outdir = tmp_path / "models"
    (outdir / "checkpoint-10").mkdir(parents=True)
    (outdir / "checkpoint-20").mkdir(parents=True)
    checkpoint = outdir / "checkpoint-20"
    for name in ("config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        (checkpoint / name).touch()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 20}))
    assert resolve_resume_checkpoint(str(outdir), True).endswith("checkpoint-20")


def test_invalid_resume_is_not_fresh_initialization(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory not found"):
        resolve_resume_checkpoint(str(tmp_path), str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="Incomplete resume checkpoint"):
        resolve_resume_checkpoint(str(tmp_path), str(tmp_path))


def test_partial_latest_and_corrupt_state_fail_visibly(tmp_path):
    older = tmp_path / "checkpoint-1"
    older.mkdir()
    for name in ("config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        (older / name).touch()
    (older / "trainer_state.json").write_text('{"global_step": 1}')
    (tmp_path / "checkpoint-2").mkdir()
    # Do not silently rewind past a partially written latest checkpoint.
    with pytest.raises(ValueError, match="Incomplete resume checkpoint"):
        resolve_resume_checkpoint(str(tmp_path), True)
    assert resolve_resume_checkpoint(str(tmp_path), str(older)) == str(older)
    (older / "trainer_state.json").write_text('{"global_step": "broken"}')
    with pytest.raises(ValueError, match="Invalid trainer_state"):
        resolve_resume_checkpoint(str(tmp_path), str(older))
