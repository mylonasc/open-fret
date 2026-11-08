import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_model import TabSampleCallback
from open_fret.viewer import renderer_for_input


class Tokenizer:
    def __call__(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    def decode(self, *args, **kwargs):
        return "TAB_1_0 TS_480 TAB_2_0 TS_480 TAB_3_0 TS_480 TAB_6_0 TS_480 " * 8


class Model(torch.nn.Module):
    device = torch.device("cpu")

    def generate(self, **kwargs):
        return torch.tensor([[1, 2]])


def generate_fixture(root):
    samples = [{"name": f"song-{n}", "tempo": 90,
                "midi_tokens": ["CAPO_2", "TUNING_64_59_55_50_45_38", "NO_66", "TS_480", "NF_66"],
                "tab_tokens": ["TAB_1_0", "TS_480"]} for n in range(5)]
    callback = TabSampleCallback(Tokenizer(), samples, str(root), "fixture")
    model = Model()
    callback.on_evaluate(None, SimpleNamespace(global_step=1, epoch=0.1), None,
                         metrics={"eval_loss": 1.6, "eval_ce_loss": 2, "eval_pitch_loss": 1}, model=model)
    checkpoint = root / "checkpoint-1"
    checkpoint.mkdir()
    callback.on_save(None, SimpleNamespace(global_step=1, epoch=0.1), None, model=model)
    callback.on_save(None, SimpleNamespace(global_step=2, epoch=0.2), None, model=model)
    return root / "samples/step-00000001/viewer.html"


def test_auto_validation_and_checkpoint_html(tmp_path):
    page = generate_fixture(tmp_path)
    assert page.exists()
    assert page.read_text().count('class="ofv-root"') == 5
    assert "eval_pitch_loss" in page.read_text()
    assert "step-00000001/viewer.html" in (tmp_path / "checkpoint-1/viewer.html").read_text()
    assert "step-00000002/viewer.html" in (tmp_path / "samples/index.html").read_text()
    manifest = json.loads((page.parent / "manifest.json").read_text())
    assert all("error" not in song for song in manifest["songs"])


def test_renderer_uses_source_tuning_and_capo():
    renderer = renderer_for_input(["CAPO_2", "TUNING_64_59_55_50_45_38"], tempo=90)
    tex = renderer.to_alphatex("TAB_1_0 TS_480 TAB_6_0")
    assert "\\tempo 90" in tex
    assert "\\tuning F#4 C#4 A3 E3 B2 E2" in tex
