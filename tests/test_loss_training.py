"""Real tiny T5 training, evaluation, save/reload and optimizer-state resume."""

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.loss_training import (TabLossTrainer, add_loss_arguments,
                                     loss_config_from_args, validate_loss_cache)
from open_fret.losses import LossConfig
from open_fret.alphatex import PARSER_VERSION


TOKENS = ("[PAD]", "[UNK]", "[BOS]", "[EOS]", "CAPO_0", "CAPO_2",
          "TUNING_64_59_55_50_45_40", "TUNING_64_59_55_50_45_38",
          "TAB_1_0", "TAB_2_5", "TAB_3_9", "TS_480", "NO_64")


class Tokenizer:
    eos_token_id = 3
    def get_vocab(self):
        return {token: i for i, token in enumerate(TOKENS)}


def model():
    return T5ForConditionalGeneration(T5Config(
        vocab_size=len(TOKENS), d_model=16, d_ff=32, num_layers=1,
        num_decoder_layers=1, num_heads=2, d_kv=8, dropout_rate=0,
        pad_token_id=0, eos_token_id=3, decoder_start_token_id=0))


def examples():
    return [{"input_ids": [4, 6, 12, 3], "attention_mask": [1, 1, 1, 1],
             "labels": [8, 11, 9, 3, -100]} for _ in range(4)]


def arguments(path, steps):
    return TrainingArguments(
        output_dir=str(path), use_cpu=True, max_steps=steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=2,
        gradient_accumulation_steps=2, learning_rate=1e-3,
        save_strategy="steps", save_steps=1, eval_strategy="steps", eval_steps=1,
        logging_steps=1, report_to="none", disable_tqdm=True, save_total_limit=3)


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("config", [LossConfig(pitch_alpha=0.3), LossConfig(smoothing=0.1),
                                    LossConfig(ce_weight=0.3, pitch_alpha=1, smoothing=0.1),
                                    LossConfig(pitch_alpha=1, smoothing=0.1, type_alpha=0.3,
                                               conditional_pitch=True)])
def test_real_training_evaluation_reload_and_resume(tmp_path, config):
    trainer = TabLossTrainer(model=model(), args=arguments(tmp_path, 1),
                             train_dataset=examples(), eval_dataset=examples(),
                             loss_tokenizer=Tokenizer(), loss_config=config)
    result = trainer.train()
    assert result.global_step == 1
    assert trainer.optimizer.state
    evaluation = trainer.evaluate()
    assert all(torch.isfinite(torch.tensor(v)) for v in evaluation.values() if isinstance(v, float))
    assert "eval_ce_loss" in evaluation
    assert evaluation["eval_note_positions"] == 8
    if config.conditional_pitch:
        # All three playable TAB tokens in this fixture have pitch 64.
        assert abs(evaluation["eval_pitch_loss"]) < 1e-6
    else:
        assert evaluation["eval_pitch_loss"] > 0 if config.pitch_alpha else evaluation["eval_smoothed_note_positions"] == 8
    checkpoint = tmp_path / "checkpoint-1"
    saved = json.loads((checkpoint / "config.json").read_text())
    assert saved["open_fret_loss"] == config.to_dict()
    assert saved["open_fret_token_metadata"]["sha256"] == trainer.token_metadata.manifest()["sha256"]
    reloaded = T5ForConditionalGeneration.from_pretrained(checkpoint)
    generated = reloaded.generate(torch.tensor([[4, 6, 12, 3]]), max_new_tokens=5)
    assert generated.ndim == 2
    resumed = TabLossTrainer(model=model(), args=arguments(tmp_path, 2),
                             train_dataset=examples(), eval_dataset=examples(),
                             loss_tokenizer=Tokenizer(), loss_config=config)
    assert resumed.train(resume_from_checkpoint=str(checkpoint)).global_step == 2
    assert max(int(s["step"]) for s in resumed.optimizer.state.values()) == 2
    assert resumed.lr_scheduler.last_epoch == 2
    assert "ce_loss" in next(row for row in resumed.state.log_history if "loss" in row)


def test_ce_default_delegates_exactly_to_hf(tmp_path):
    tiny = model()
    trainer = TabLossTrainer(model=tiny, args=arguments(tmp_path, 1),
                             loss_tokenizer=Tokenizer(), eval_dataset=examples())
    baseline = Trainer(model=tiny, args=arguments(tmp_path / "baseline", 1), eval_dataset=examples())
    batch = {k: torch.tensor([v]) for k, v in examples()[0].items()}
    actual = trainer.compute_loss(tiny, dict(batch))
    expected = baseline.compute_loss(tiny, dict(batch))
    assert torch.equal(actual, expected)


def test_source_conditioning_survives_hf_column_pruning(tmp_path):
    from open_fret.token_metadata import MAX_STRINGS
    rows = examples()
    for row in rows:
        row["input_ids"][1] = 1  # unknown composite tuning, not a standard-tuning fallback
        row["loss_tuning"] = [64, 59, 55, 50, 45, 40] + [-1] * (MAX_STRINGS - 6)
        row["loss_capo"] = 0
    trainer = TabLossTrainer(model=model(), args=arguments(tmp_path, 1), train_dataset=rows,
                             eval_dataset=rows, loss_tokenizer=Tokenizer(),
                             loss_config=LossConfig(ce_weight=0.3, pitch_alpha=1, smoothing=0.1))
    assert trainer.train().global_step == 1
    assert trainer.evaluate()["eval_weighted_ce_loss"] > 0


def test_resume_rejects_changed_loss_and_metadata(tmp_path):
    trainer = TabLossTrainer(model=model(), args=arguments(tmp_path, 1),
                             loss_tokenizer=Tokenizer(), loss_config=LossConfig(pitch_alpha=0.3),
                             eval_dataset=examples())
    trainer.save_model(str(tmp_path / "checkpoint"))
    other = TabLossTrainer(model=model(), args=arguments(tmp_path / "other", 1),
                           loss_tokenizer=Tokenizer(), loss_config=LossConfig(smoothing=0.1),
                           eval_dataset=examples())
    with pytest.raises(ValueError, match="Checkpoint loss differs"):
        other._load_from_checkpoint(str(tmp_path / "checkpoint"))
    config_path = tmp_path / "checkpoint" / "config.json"
    saved = json.loads(config_path.read_text())
    saved["open_fret_token_metadata"]["sha256"] = "wrong"
    config_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="metadata differs"):
        trainer._load_from_checkpoint(str(tmp_path / "checkpoint"))


def test_cli_default_and_separate_ablations():
    parser = argparse.ArgumentParser()
    add_loss_arguments(parser)
    assert not loss_config_from_args(parser.parse_args([])).enabled
    assert loss_config_from_args(parser.parse_args(["--pitch-loss-alpha", "0.3"])).pitch_alpha == 0.3
    assert loss_config_from_args(parser.parse_args(["--equivalence-smoothing", "0.1"])).smoothing == 0.1
    combined = loss_config_from_args(parser.parse_args([
        "--ce-loss-weight", "0.3", "--pitch-loss-alpha", "1", "--equivalence-smoothing", "0.1"]))
    assert combined == LossConfig(ce_weight=0.3, pitch_alpha=1, smoothing=0.1)


def test_custom_loss_rejects_legacy_cache_without_changing_it(tmp_path):
    path = tmp_path / "cache_manifest.json"
    validate_loss_cache(str(tmp_path), LossConfig())
    for content in [None, {"splits": {"train": 4}}, {"parser_version": "old"}]:
        if content is not None:
            path.write_text(json.dumps(content))
        with pytest.raises(ValueError, match="obsolete/unversioned"):
            validate_loss_cache(str(tmp_path), LossConfig(pitch_alpha=0.3))
        if content is not None:
            assert json.loads(path.read_text()) == content
    path.write_text(json.dumps({"parser_version": PARSER_VERSION}))
    validate_loss_cache(str(tmp_path), LossConfig(smoothing=0.1))
