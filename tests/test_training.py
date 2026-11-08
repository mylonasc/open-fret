import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.training import resolve_resume_checkpoint


def test_fresh_start_returns_none(tmp_path):
    outdir = tmp_path / "models"
    outdir.mkdir()
    assert resolve_resume_checkpoint(str(outdir), True) is None
    assert resolve_resume_checkpoint(str(outdir), False) is None
    assert resolve_resume_checkpoint(str(outdir), None) is None


def test_resume_finds_latest(tmp_path):
    outdir = tmp_path / "models"
    (outdir / "checkpoint-10").mkdir(parents=True)
    (outdir / "checkpoint-20").mkdir(parents=True)
    assert resolve_resume_checkpoint(str(outdir), True).endswith("checkpoint-20")
