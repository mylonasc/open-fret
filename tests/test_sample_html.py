import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_model import TabSampleCallback
from open_fret.viewer import renderer_for_input


def real_model_and_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration
    words = ['[PAD]', '[UNK]', '[EOS]', 'CAPO_2', 'TUNING_64_59_55_50_45_38',
             'NO_66', 'NF_66', 'TS_480', 'TAB_1_0', 'TAB_2_5']
    backend = Tokenizer(WordLevel({t: i for i, t in enumerate(words)}, unk_token='[UNK]'))
    backend.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                 eos_token='[EOS]', unk_token='[UNK]')
    model = T5ForConditionalGeneration(T5Config(vocab_size=len(words), d_model=16, d_ff=32,
        num_layers=1, num_decoder_layers=1, num_heads=2, pad_token_id=0,
        eos_token_id=2, decoder_start_token_id=0))
    return tok, model


def generate_fixture(root):
    samples = [{"name": f"song-{n}", "tempo": 90,
                "midi_tokens": ["CAPO_2", "TUNING_64_59_55_50_45_38", "NO_66", "TS_480", "NF_66"],
                "tab_tokens": ["TAB_1_0", "TS_480"]} for n in range(5)]
    tok, model = real_model_and_tokenizer()
    callback = TabSampleCallback(tok, samples, str(root), "fixture")
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


def test_select_sample_songs_skips_unsupported_midi(tmp_path):
    from scripts.train_model import select_sample_songs
    tok, _ = real_model_and_tokenizer()
    data = tmp_path / "individual"
    data.mkdir()
    rows = {
        "bad.tex__t1__c2__w0.json": ["CAPO_2", "TUNING_64_59_55_50_45_38", "NO_20"],
        "good1.tex__t1__c2__w0.json": ["CAPO_2", "TUNING_64_59_55_50_45_38", "NO_66"],
        "good2.tex__t1__c2__w0.json": ["CAPO_2", "TUNING_64_59_55_50_45_38", "NO_64"],
    }
    for name, midi in rows.items():
        (data / name).write_text(json.dumps({"midi": midi, "tab": ["TAB_1_0"]}))
    split = tmp_path / "val_files.txt"
    split.write_text("\n".join(rows))
    picked = select_sample_songs(str(split), str(data), k=1, tokenizer=tok)
    assert picked[0]["name"] == "good1.tex__t1__c2__w0"


def test_renderer_uses_source_tuning_and_capo():
    renderer = renderer_for_input(["CAPO_2", "TUNING_64_59_55_50_45_38"], tempo=90)
    tex = renderer.to_alphatex("TAB_1_0 TS_480 TAB_6_0")
    assert "\\tempo 90" in tex
    assert "\\tuning F#4 C#4 A3 E3 B2 E2" in tex
