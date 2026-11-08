"""GPU subset validation: vocab -> tokenizer -> cache -> short training.

Validates that training works on real data (from scratch and from a
pretrained T5 endpoint) without waiting for the full-dataset pipeline.
"""
import glob
import json
import os
import random
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

WORK = "/tmp/opencode/gpu_subset"
N_TRAIN, N_VAL = 4000, 500


def main():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (PreTrainedTokenizerFast, T5ForConditionalGeneration,
                              Trainer, TrainingArguments, set_seed)
    from datasets import Dataset
    from config import get_tiny_tab_config
    from open_fret.training import load_init_model

    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    all_files = sorted(glob.glob("dataset/processed/individual/*.json"))
    # Exclude the failures log (not .json, already excluded).
    rnd = random.Random(0)
    sample = rnd.sample(all_files, N_TRAIN + N_VAL)
    train_files, val_files = sample[:N_TRAIN], sample[N_TRAIN:]

    # Vocab from subset train files.
    vocab_set = set()
    for f in train_files:
        d = json.load(open(f))
        vocab_set.update(d["midi"])
        vocab_set.update(d["tab"])
    specials = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
    vocab = specials + sorted(vocab_set)
    tok = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(vocab)},
                                     unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(os.path.join(WORK, "tokenizer.json"))
    fast = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(WORK, "tokenizer.json"),
        bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
        pad_token="[PAD]", mask_token="[MASK]")
    tokdir = os.path.join(WORK, "tok")
    fast.save_pretrained(tokdir)
    print(f"subset vocab={len(fast)}", flush=True)

    def records(files):
        out = []
        for f in files:
            d = json.load(open(f))
            out.append({"midi_text": " ".join(d["midi"]) + " " + fast.eos_token,
                        "tab_text": " ".join(d["tab"]) + " " + fast.eos_token})
        return out

    def tok_fn(ex):
        mi = fast(text=ex["midi_text"], text_target=ex["tab_text"],
                  max_length=512, truncation=True, padding="max_length")
        mi["labels"] = [[t if t != fast.pad_token_id else -100 for t in s]
                        for s in mi["labels"]]
        return mi

    train_ds = Dataset.from_list(records(train_files)).map(
        tok_fn, batched=True, remove_columns=["midi_text", "tab_text"])
    val_ds = Dataset.from_list(records(val_files)).map(
        tok_fn, batched=True, remove_columns=["midi_text", "tab_text"])

    # A) from scratch, 50 steps.
    set_seed(0)
    model, info = load_init_model(fast, init_from=None,
                                  tiny_config_fn=get_tiny_tab_config)
    print("init A:", info, flush=True)
    args = TrainingArguments(
        output_dir=os.path.join(WORK, "out_scratch"),
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        max_steps=50, learning_rate=5e-4, warmup_steps=10,
        lr_scheduler_type="cosine", optim="adafactor",
        eval_strategy="steps", eval_steps=25, save_strategy="no",
        logging_strategy="steps", logging_steps=10, report_to="none", seed=0)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds)
    history = trainer.train()
    print(f"scratch 50 steps done. final loss={history.training_loss:.4f}", flush=True)
    print("eval:", trainer.evaluate(), flush=True)
    assert history.training_loss < 10.0, "scratch training diverged"

    # B) pretrained endpoint init + a few steps (proves --init-from path).
    try:
        model_b, info_b = load_init_model(fast, init_from="t5-small")
        print("init B:", {k: v for k, v in info_b.items() if k != "aligned_ids"},
              flush=True)
        args_b = TrainingArguments(
            output_dir=os.path.join(WORK, "out_pre"),
            per_device_train_batch_size=2, per_device_eval_batch_size=2,
            max_steps=5, learning_rate=1e-4, warmup_steps=2,
            optim="adafactor", eval_strategy="no", save_strategy="no",
            logging_strategy="no", report_to="none", seed=0)
        trainer_b = Trainer(model=model_b, args=args_b, train_dataset=train_ds,
                            eval_dataset=val_ds)
        trainer_b.train()
        print("pretrained t5-small init + 5 steps OK", flush=True)
    except Exception as e:
        print(f"pretrained path FAILED: {e}")
        raise
    print("GPU SUBSET VALIDATION PASSED")


if __name__ == "__main__":
    main()
