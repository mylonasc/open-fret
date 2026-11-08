"""Real T5 cache/beam tests; no mocked generation logits."""
import itertools
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_fret.generation import (AnchoredTemplate, UnsupportedInputError, UnplayableInputError,
                                  chunk_complete_onsets, generate_tab, prepare_encoder)
from open_fret.transcription_diagnostics import compare_streams


@pytest.fixture
def setup():
    words = ['[PAD]', '[UNK]', '[EOS]', 'CAPO_0', 'CAPO_2', 'TUNING_64_59',
             'NO_64', 'NO_66', 'NF_64', 'NF_66', 'NO_59', 'TS_120', 'TS_0',
             'TAB_1_0', 'TAB_2_5', 'TAB_1_0_b4', 'TAB_2_5_b4',
             'TAB_1_2', 'TAB_2_7', 'TAB_2_0']
    base = Tokenizer(WordLevel({t: i for i, t in enumerate(words)}, unk_token='[UNK]'))
    base.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=base, pad_token='[PAD]',
                                 unk_token='[UNK]', eos_token='[EOS]')
    torch.manual_seed(42)
    model = T5ForConditionalGeneration(T5Config(vocab_size=len(words), d_model=16,
        d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.pad_token_id)).eval()
    return tok, model


PREFIX = ['CAPO_0', 'TUNING_64_59']


def test_cached_beam_matches_exhaustive_uncached_scoring(setup):
    tok, model = setup
    tokens = PREFIX + 'NO_64 NO_66 TS_120 NF_64 NF_66'.split()
    template = AnchoredTemplate(tok, tokens)
    paths = [[model.config.decoder_start_token_id]]
    for position in range(len(template.slots)):
        paths = [p + [i] for p in paths for i in template.allowed(position, p)]
    encoded = prepare_encoder(tok, tokens)
    scores = []
    with torch.no_grad():
        for p in paths:
            labels = torch.tensor([p[1:]])
            logp = model(**encoded, labels=labels).logits.log_softmax(-1)
            scores.append(float(logp.gather(-1, labels.unsqueeze(-1)).sum()))
    best = paths[max(range(len(scores)), key=scores.__getitem__)]
    actual = generate_tab(model, tok, tokens, beams=len(paths))
    assert actual.tokens == tok.decode(best, skip_special_tokens=True).split()
    assert actual.stopped_eos
    assert not model.training


def test_chord_lookahead_preserves_only_string_for_later_note(setup):
    tok, model = setup
    # First note can use either string; pitch 59 can only use string two.
    tokens = PREFIX + 'NO_64 NO_59 TS_120'.split()
    template = AnchoredTemplate(tok, tokens)
    assert {template.note_strings[i] for i in template.allowed(0, [0])} == {1}
    result = generate_tab(model, tok, tokens, beams=4)
    assert [t.split('_')[1] for t in result.tokens if t.startswith('TAB_')] == ['1', '2']


def test_technique_variants_do_not_corrupt_string_bitmask(setup):
    tok, model = setup
    tokens = PREFIX + 'NO_64 NO_64 TS_120'.split()
    result = generate_tab(model, tok, tokens, beams=8)
    notes = [t for t in result.tokens if t.startswith('TAB_')]
    assert len(notes) == 2 and len({t.split('_')[1] for t in notes}) == 2
    with pytest.raises(UnplayableInputError):
        AnchoredTemplate(tok, PREFIX + ['NO_64'] * 3)


def test_repeated_riff_timing_and_capo(setup):
    tok, model = setup
    tokens = ['CAPO_2', 'TUNING_64_59'] + ('NO_66 TS_120 NF_66 TS_120 '.split() * 5)
    model.train()
    result = generate_tab(model, tok, tokens)
    assert model.training
    metrics = compare_streams(tokens, result.tokens, result.tokens)
    assert metrics['input_reference_onsets_identical']
    assert metrics['timed_pitch']['f1'] == 1
    assert metrics['prediction']['same_string_conflict_onsets'] == 0


def test_budget_unknown_and_eos_fail_without_silent_truncation(setup):
    tok, model = setup
    tokens = PREFIX + ['NO_64']
    assert prepare_encoder(tok, tokens)['input_ids'][0, -1] == tok.eos_token_id
    assert torch.equal(prepare_encoder(tok, tokens)['input_ids'],
                       prepare_encoder(tok, tokens + [tok.eos_token])['input_ids'])
    for bad in (tokens + ['TS_999'], tokens + ['[UNK]'], PREFIX + ['[EOS]', 'NO_64']):
        with pytest.raises(UnsupportedInputError):
            prepare_encoder(tok, bad)
    with pytest.raises(UnsupportedInputError):
        generate_tab(model, tok, tokens, max_input_tokens=3)
    with pytest.raises(UnsupportedInputError):
        generate_tab(model, tok, tokens, max_new_tokens=1)


def test_chunking_never_divides_same_tick_chord():
    tokens = 'NO_64 NO_66 TS_120 NF_64 NF_66 NO_64 NO_66 TS_120 NF_64 NF_66'.split()
    chunks = chunk_complete_onsets(tokens, 5)
    assert list(itertools.chain.from_iterable(chunks)) == tokens
    assert all(len(c) <= 5 for c in chunks)
    assert chunks[1] == 'TS_120 NF_64 NF_66 NO_64 NO_66'.split()
    with pytest.raises(UnsupportedInputError):
        chunk_complete_onsets(tokens, 4)
