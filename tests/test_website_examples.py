"""The website example builder emits self-contained playable pages."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_website_examples_build(tmp_path):
    static_root = tmp_path / "examples"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_website_examples.py"),
         "--static-root", str(static_root)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stderr
    slugs = {"crash", "the-way-i-feel", "pahilo-junema", "glory", "end-games"}
    assert {p.name for p in static_root.iterdir() if p.is_dir()} == slugs
    for slug in sorted(slugs):
        page = (static_root / slug / "index.html").read_text()
        assert "alphaTab" in page and "wave-roll" in page
        assert "ofv-" in page
        meta = json.loads((static_root / slug / "meta.json").read_text())
        assert meta["n_input"] > 0 and meta["n_pred"] > 0
        assert meta["tempo"] > 0 and meta["title"]
