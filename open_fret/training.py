"""Training helpers: resume handling and model/tokenizer wiring (Issue #3)."""

from __future__ import annotations

import os


def find_latest_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
            except (IndexError, ValueError):
                continue
            candidates.append((step, full))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def resolve_resume_checkpoint(output_dir: str, resume: str | bool | None) -> str | bool | None:
    """Resolve what to pass as ``resume_from_checkpoint``.

    - ``False``/``None`` -> start fresh (None).
    - ``True`` -> latest checkpoint in output_dir, or None if none exists
      (fresh start instead of crashing).
    - str path -> that path (validated by caller/Trainer).
    """
    if resume is True:
        return find_latest_checkpoint(output_dir)
    if resume in (False, None):
        return None
    return resume


def load_init_model(tokenizer, init_from: str | None = None,
                    tiny_config_fn=None, dropout: float | None = None):
    """Build the starting model for training.

    - ``init_from=None``: random tiny-tab init via ``tiny_config_fn``.
    - ``init_from=<hub id or local path>``: load pretrained
      ``T5ForConditionalGeneration`` weights, resize embeddings when the
      custom tokenizer size differs, and align pad/eos/bos/unk +
      decoder-start ids to the tokenizer.

    Returns ``(model, info)`` where info records the init mode for logging.
    """
    if init_from is None:
        if tiny_config_fn is None:
            raise ValueError("tiny_config_fn is required when init_from is None.")
        config = tiny_config_fn(vocab_size=len(tokenizer))
        if dropout is not None:
            config.dropout_rate = dropout
        from transformers import T5ForConditionalGeneration
        return T5ForConditionalGeneration(config), {
            "mode": "scratch", "init_from": None,
            "vocab_size": len(tokenizer),
        }
    from transformers import T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained(init_from)
    base_vocab = model.config.vocab_size
    target_vocab = len(tokenizer)
    resized = False
    if base_vocab != target_vocab:
        model.resize_token_embeddings(target_vocab)
        resized = True
    # Align generation/special ids to OUR tokenizer (pretrained T5 uses
    # eos=1 etc., ours uses pad=0/eos=3/bos=2/unk=1).
    updates = {}
    for attr, tok_attr in (("pad_token_id", "pad_token_id"),
                           ("eos_token_id", "eos_token_id"),
                           ("bos_token_id", "bos_token_id"),
                           ("unk_token_id", "unk_token_id")):
        tok_id = getattr(tokenizer, tok_attr, None)
        if tok_id is not None:
            setattr(model.config, attr, tok_id)
            updates[attr] = tok_id
    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        updates["decoder_start_token_id"] = tokenizer.pad_token_id
    model.config.vocab_size = target_vocab
    if dropout is not None:
        model.config.dropout_rate = dropout
    return model, {"mode": "pretrained", "init_from": init_from,
                   "base_vocab_size": base_vocab, "vocab_size": target_vocab,
                   "resized_embeddings": resized, "aligned_ids": updates}


def training_layout(output_dir: str) -> dict[str, str]:
    return {
        "output_dir": output_dir,
        "final_dir": os.path.join(output_dir, "final"),
    }
