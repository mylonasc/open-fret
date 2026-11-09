import sys
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.training import load_init_model, resolve_resume_checkpoint


class FakeTokenizer:
    def __init__(self, size, pad=0, eos=3, bos=2, unk=1):
        self._size = size
        self.pad_token_id = pad
        self.eos_token_id = eos
        self.bos_token_id = bos
        self.unk_token_id = unk

    def __len__(self):
        return self._size


def test_scratch_init_uses_tiny_config():
    from scripts.config import get_tiny_tab_config
    tok = FakeTokenizer(123)
    model, info = load_init_model(tok, init_from=None,
                                  tiny_config_fn=get_tiny_tab_config, dropout=0.05)
    assert info["mode"] == "scratch"
    assert model.config.vocab_size == 123
    assert model.config.dropout_rate == 0.05


def test_pretrained_init_resizes_and_aligns_ids(tmp_path):
    from transformers import T5ForConditionalGeneration
    from scripts.config import get_tiny_tab_config
    # Fake "pretrained" endpoint: tiny random T5 saved locally.
    base_tok = FakeTokenizer(64)
    base_model, _ = load_init_model(base_tok, init_from=None,
                                    tiny_config_fn=get_tiny_tab_config)
    base_model.config.pad_token_id = 0
    base_model.config.eos_token_id = 1  # deliberately different from target
    pretrained_dir = tmp_path / "pretrained"
    base_model.save_pretrained(pretrained_dir)
    # Target tokenizer: different size + different special ids.
    target_tok = FakeTokenizer(96, pad=0, eos=3, bos=2, unk=1)
    model, info = load_init_model(target_tok, init_from=str(pretrained_dir))
    assert info["mode"] == "pretrained"
    assert info["resized_embeddings"] is True
    assert model.config.vocab_size == 96
    assert model.config.eos_token_id == 3
    assert model.config.decoder_start_token_id == 0
    assert model.get_input_embeddings().weight.shape[0] == 96


def test_resolve_resume_none_for_fresh(tmp_path):
    assert resolve_resume_checkpoint(str(tmp_path), None) is None
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        resolve_resume_checkpoint(str(tmp_path), True)
