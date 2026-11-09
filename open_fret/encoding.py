"""One strict cached/uncached encoding contract; never truncate supervision."""
from .curation import conflict_positions
from .token_metadata import MAX_STRINGS, TokenMetadata


def encode_window(data, tokenizer, max_length=512, mask_conflicts=True):
    midi, tab = data['midi'], data['tab']
    if (not isinstance(midi, list) or not isinstance(tab, list)
            or not midi or not tab or not all(isinstance(t, str) for t in midi + tab)):
        raise ValueError("Expected nonempty midi/tab token lists")
    if len(midi) + 1 > max_length or len(tab) + 1 > max_length:
        raise ValueError("Window exceeds token budget including EOS; re-window, never truncate")
    vocab = tokenizer.get_vocab()
    if not tokenizer.eos_token:
        raise ValueError("Tokenizer must define EOS")
    unknown = [t for t in midi + tab if t not in vocab or t in tokenizer.all_special_tokens]
    if unknown:
        raise ValueError(f"Unknown/special domain tokens: {unknown[:3]}")
    tuning = data.get('tuning') or TokenMetadata.tuning(midi[1])
    capo = data.get('capo', TokenMetadata.capo(midi[0]))
    if tuning is None or capo is None or len(tuning) > MAX_STRINGS:
        raise ValueError("Missing/unsupported source conditioning")
    ids, labels = [], []
    for tokens, result in ((midi, ids), (tab, labels)):
        text = ' '.join(tokens + [tokenizer.eos_token])
        encoded = tokenizer.encode(text, add_special_tokens=False)
        expected = [vocab[t] for t in tokens] + [tokenizer.eos_token_id]
        if encoded != expected:
            raise ValueError("Tokenizer must encode each domain token as one ID")
        result.extend(encoded)
    attention = [1] * len(ids) + [0] * (max_length - len(ids))
    ids += [tokenizer.pad_token_id] * (max_length - len(ids))
    if mask_conflicts:
        for i in conflict_positions(tab):
            labels[i] = -100
    labels += [-100] * (max_length - len(labels))
    return {'input_ids': ids, 'attention_mask': attention, 'labels': labels,
            'loss_tuning': list(tuning) + [-1] * (MAX_STRINGS - len(tuning)), 'loss_capo': capo}
