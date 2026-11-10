import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_fret.losses import LossConfig, TabLoss
from open_fret.token_metadata import TokenMetadata


TOKENS = (
    "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "CAPO_0", "CAPO_2",
    "TUNING_64_59_55_50_45_40", "TUNING_64_59_55_50_45_38",
    "TUNING_65_59_55_50_45_40", "TUNING_64_59_55_50",
    "TS_480", "TAB_1_0", "TAB_2_5", "TAB_3_9", "TAB_1_2",
    "TAB_2_3", "TAB_6_0", "TAB_6_2", "TAB_1_0_b4", "TAB_2_5_b4",
    "TAB_0_3", "TAB_1_25", "TAB_7_0", "TAB_bad",
)
IDS = {t: i for i, t in enumerate(TOKENS)}
META = TokenMetadata(TOKENS)


def prefix(tuning="TUNING_64_59_55_50_45_40", capo="CAPO_0"):
    return torch.tensor([[IDS[capo], IDS[tuning], IDS["[EOS]"]]])


def objective(config, scores, targets, inputs=None):
    labels = torch.tensor([[IDS[t] if isinstance(t, str) else t for t in targets]])
    return TabLoss(META, config)(scores, labels, prefix() if inputs is None else inputs)


def test_metadata_concert_pitch_custom_tuning_capo_and_hash():
    standard = (64, 59, 55, 50, 45, 40)
    assert META.pitch(IDS["TAB_1_0"], standard) == 64
    assert META.pitch(IDS["TAB_2_5"], standard) == 64
    assert META.pitch(IDS["TAB_6_0"], standard, capo=2) == 42
    assert META.pitch(IDS["TAB_6_0"], (*standard[:-1], 38), capo=2) == 40
    assert META.pitch(IDS["TS_480"], standard) is None
    assert META.pitch(IDS["TAB_7_0"], standard) is None
    assert META.pitch(IDS["TAB_1_25"], standard) is None
    assert META.tuning("TUNING_43_38_33_28_-1_-1") == (43, 38, 33, 28, -1, -1)
    assert META.pitch(IDS["TAB_6_0"], (43, 38, 33, 28, -1, -1)) is None
    assert META.manifest() == TokenMetadata(TOKENS).manifest()
    assert META.manifest()["sha256"] != TokenMetadata(tuple(reversed(TOKENS))).manifest()["sha256"]


def test_conditional_pitch_matches_playable_normalization_and_ignores_structural_mass():
    torch.manual_seed(9)
    scores = torch.randn(1, 1, len(TOKENS), requires_grad=True)
    config = LossConfig(pitch_alpha=1, conditional_pitch=True)
    _, parts = objective(config, scores, ['TAB_1_0'])
    tuning = (64, 59, 55, 50, 45, 40)
    playable = [i for i in range(len(TOKENS)) if META.pitch(i, tuning) is not None]
    correct = [i for i in playable if META.pitch(i, tuning) == 64]
    expected = scores[0, 0, playable].logsumexp(0) - scores[0, 0, correct].logsumexp(0)
    assert torch.allclose(parts['pitch'], expected)
    changed = scores.detach().clone()
    changed[..., IDS['TS_480']] += 20
    _, after = objective(config, changed, ['TAB_1_0'])
    assert torch.allclose(parts['pitch'], after['pitch'], atol=2e-6)
    grad = torch.autograd.grad(expected, scores)[0]
    assert grad[..., IDS['TS_480']] == 0


def test_event_type_loss_is_macro_balanced_and_pushes_time_and_eos():
    scores = torch.zeros(1, 5, len(TOKENS), requires_grad=True)
    targets = ['TAB_1_0', 'TAB_2_5', 'TAB_3_9', 'TS_480', '[EOS]']
    loss, parts = objective(LossConfig(type_alpha=1), scores, targets)
    logp = scores.log_softmax(-1)
    tabs = [i for i in range(len(TOKENS)) if META.note(i)]
    expected = -(logp[0, :3, tabs].logsumexp(-1).mean()
                 + logp[0, 3, IDS['TS_480']] + logp[0, 4, IDS['[EOS]']]) / 3
    assert torch.allclose(parts['type'], expected)
    assert torch.allclose(loss, parts['ce'] + expected)
    loss.backward()
    assert scores.grad[0, 3, IDS['TS_480']] < 0
    assert scores.grad[0, 4, IDS['[EOS]']] < 0
    assert torch.isfinite(scores.grad).all()


def test_new_losses_all_padding_have_finite_zero_gradient():
    scores = torch.randn(1, 2, len(TOKENS), requires_grad=True)
    loss, parts = objective(LossConfig(type_alpha=1, pitch_alpha=1, conditional_pitch=True),
                            scores, [-100, -100])
    assert loss == 0
    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


@pytest.mark.parametrize("config", [LossConfig(), LossConfig(pitch_alpha=0), LossConfig(smoothing=0)])
def test_disabled_matches_ce_value_and_gradient(config):
    torch.manual_seed(0)
    scores = torch.randn(1, 3, len(TOKENS), requires_grad=True)
    loss, parts = objective(config, scores, ["TAB_1_0", "[EOS]", -100])
    expected = F.cross_entropy(scores[0, :2], torch.tensor([IDS["TAB_1_0"], IDS["[EOS]"]]))
    assert torch.equal(loss, expected)
    assert torch.equal(torch.autograd.grad(loss, scores, retain_graph=True)[0],
                       torch.autograd.grad(expected, scores)[0])
    assert parts["tokens"] == 2


def test_pitch_loss_matches_exact_marginal_and_has_gradients():
    scores = torch.randn(1, 2, len(TOKENS), requires_grad=True)
    loss, parts = objective(LossConfig(pitch_alpha=0.3), scores, ["TAB_1_0", "[EOS]"])
    equivalent = [IDS[t] for t in ("TAB_1_0", "TAB_2_5", "TAB_3_9", "TAB_1_0_b4", "TAB_2_5_b4")]
    expected_pitch = -scores[0, 0].log_softmax(-1)[equivalent].logsumexp(-1)
    assert torch.allclose(parts["pitch"], expected_pitch)
    assert torch.allclose(loss, parts["ce"] + 0.3 * expected_pitch)
    loss.backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0


def test_pitch_loss_equivalent_is_better_than_wrong_or_non_note():
    losses = []
    for prediction in ["TAB_2_5", "TAB_1_2", "[EOS]"]:
        scores = torch.full((1, 1, len(TOKENS)), -20.0)
        scores[0, 0, IDS[prediction]] = 20
        _, parts = objective(LossConfig(pitch_alpha=1), scores, ["TAB_1_0"])
        losses.append(float(parts["pitch"]))
    assert losses[0] < 1e-5 and losses[1] > 30 and losses[2] > 30


def test_wrong_pitches_cannot_cancel_to_correct_mean():
    scores = torch.full((1, 1, len(TOKENS)), -20.0)
    # 62 and 66 average to 64, but neither sounds the target E4.
    scores[0, 0, [IDS["TAB_2_3"], IDS["TAB_1_2"]]] = 20
    _, parts = objective(LossConfig(pitch_alpha=1), scores, ["TAB_1_0"])
    assert parts["pitch"] > 30


def test_custom_tuning_changes_equivalence_and_capo_is_supported():
    scores = torch.full((1, 1, len(TOKENS)), -10.0)
    scores[0, 0, IDS["TAB_2_5"]] = 10
    _, standard = objective(LossConfig(pitch_alpha=1), scores, ["TAB_1_0"])
    _, custom = objective(LossConfig(pitch_alpha=1), scores, ["TAB_1_0"],
                          prefix("TUNING_65_59_55_50_45_40", "CAPO_2"))
    _, capo = objective(LossConfig(pitch_alpha=1), scores, ["TAB_1_0"], prefix(capo="CAPO_2"))
    assert custom["pitch"] > standard["pitch"] + 10
    assert torch.equal(capo["pitch"], standard["pitch"])


def test_smoothing_is_normalized_same_pitch_soft_ce():
    scores = torch.randn(1, 2, len(TOKENS), requires_grad=True)
    loss, parts = objective(LossConfig(smoothing=0.2), scores, ["TAB_1_0", "[EOS]"])
    q = torch.zeros(2, len(TOKENS))
    q[0, IDS["TAB_1_0"]] = 0.8
    q[0, [IDS["TAB_2_5"], IDS["TAB_3_9"]]] = 0.1
    q[1, IDS["[EOS]"]] = 1
    assert torch.allclose(q.sum(-1), torch.ones(2))
    expected = -(q * scores[0].log_softmax(-1)).sum(-1).mean()
    assert torch.allclose(loss, expected)
    assert parts["smoothed_notes"] == 1
    assert torch.allclose(torch.autograd.grad(loss, scores, retain_graph=True)[0],
                          torch.autograd.grad(expected, scores)[0])


@pytest.mark.parametrize("target", ["TAB_1_0_b4", "TAB_6_0", "TS_480", "[EOS]"])
def test_smoothing_keeps_techniques_non_notes_and_no_alternative_hard(target):
    scores = torch.randn(1, 1, len(TOKENS))
    loss, parts = objective(LossConfig(smoothing=0.2), scores, [target])
    assert torch.equal(loss, parts["ce"])
    assert parts["smoothed_notes"] == 0


@pytest.mark.parametrize("config", [LossConfig(pitch_alpha=0.3), LossConfig(smoothing=0.1)])
def test_padding_no_notes_and_all_ignored_are_finite(config):
    scores = torch.randn(1, 3, len(TOKENS), requires_grad=True)
    loss, parts = objective(config, scores, ["TS_480", "[EOS]", -100])
    assert torch.equal(loss, parts["ce"]) and parts["notes"] == 0
    loss.backward()
    assert scores.grad[0, 2].eq(0).all()
    loss, parts = objective(config, scores, [-100, -100, -100])
    assert loss == 0 and torch.isfinite(loss)
    assert parts["pitch"] == 0 and parts["smoothed_notes"] == 0
    loss.backward()


@pytest.mark.parametrize("target", ["TAB_0_3", "TAB_1_25", "TAB_7_0", "TAB_bad", -1, 10000])
def test_bad_supervision_rejected(target):
    with pytest.raises(ValueError):
        objective(LossConfig(pitch_alpha=1), torch.randn(1, 1, len(TOKENS)), [target])


def test_missing_conditioning_and_string_outside_tuning_rejected():
    scores = torch.randn(1, 1, len(TOKENS))
    with pytest.raises(ValueError, match="Expected CAPO"):
        objective(LossConfig(pitch_alpha=1), scores, ["TAB_1_0"], torch.tensor([[0, 0]]))
    with pytest.raises(ValueError, match="outside example"):
        objective(LossConfig(pitch_alpha=1), scores, ["TAB_6_0"], prefix("TUNING_64_59_55_50"))


def test_unavailable_strings_are_not_candidates_or_targets():
    tokens = (*TOKENS, "TUNING_64_59_55_50_-1_-1")
    metadata = TokenMetadata(tokens)
    module = TabLoss(metadata, LossConfig(pitch_alpha=0.3))
    inputs = torch.tensor([[IDS["CAPO_0"], len(tokens) - 1]])
    scores = torch.randn(1, 1, len(tokens), requires_grad=True)
    loss, _ = module(scores, torch.tensor([[IDS["TAB_1_0"]]]), inputs)
    assert torch.isfinite(loss)
    loss.backward()
    with pytest.raises(ValueError, match="outside example"):
        module(scores, torch.tensor([[IDS["TAB_6_0"]]]), inputs)


@pytest.mark.parametrize("kwargs", [
    {"pitch_alpha": -1}, {"pitch_alpha": math.nan}, {"smoothing": 1},
    {"smoothing": -0.1}, {"smoothing": math.inf}, {"ce_weight": -1},
    {"position_alpha": -0.5}, {"span_alpha": -1}, {"stretch_alpha": math.nan},
    {"span_margin": -1}, {"stretch_margin": 2.5},
])
def test_bad_loss_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        LossConfig(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("config", [LossConfig(pitch_alpha=0.3), LossConfig(smoothing=0.1)])
def test_mixed_precision_extreme_logits_stable(dtype, config):
    scores = torch.full((1, 1, len(TOKENS)), -10000.0, dtype=dtype)
    scores[0, 0, IDS["TAB_1_2"]] = 10000
    scores.requires_grad_()
    loss, _ = objective(config, scores, ["TAB_1_0"])
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert torch.isfinite(scores.grad).all()


def test_mixed_conditioning_batch_matches_individual_examples():
    scores = torch.randn(2, 2, len(TOKENS), requires_grad=True)
    labels = torch.tensor([[IDS["TAB_1_0"], IDS["TAB_2_5"]]] * 2)
    inputs = torch.cat([prefix(), prefix("TUNING_65_59_55_50_45_40", "CAPO_2")])
    for config in [LossConfig(pitch_alpha=0.3), LossConfig(smoothing=0.1)]:
        loss, _ = TabLoss(META, config)(scores, labels, inputs)
        individual = [TabLoss(META, config)(scores[i:i+1], labels[i:i+1], inputs[i:i+1])[0]
                      for i in range(2)]
        assert torch.allclose(loss, torch.stack(individual).mean())


def test_weighted_combined_objective_and_non_note_batches():
    scores = torch.randn(1, 2, len(TOKENS), requires_grad=True)
    config = LossConfig(ce_weight=0.3, pitch_alpha=1, smoothing=0.1)
    loss, parts = objective(config, scores, ["TAB_1_0", "[EOS]"])
    assert torch.allclose(loss, 0.3 * (parts["ce"] + parts["smoothing_delta"]) + parts["pitch"])
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    loss, parts = objective(config, scores, ["TS_480", "[EOS]"])
    assert torch.allclose(loss, 0.3 * parts["ce"])


def _one_hot_scores(token, val=20.0):
    scores = torch.full((1, 1, len(TOKENS)), -20.0)
    scores[0, 0, IDS[token]] = val
    return scores


def test_positional_matches_expected_fret_distance():
    # Target TAB_2_5 (fret 5); all mass on TAB_2_3 (same string, fret 3).
    _, parts = objective(LossConfig(position_alpha=1.0),
                         _one_hot_scores("TAB_2_3"), ["TAB_2_3"])
    assert parts["position"] == 0.0
    _, parts = objective(LossConfig(position_alpha=1.0),
                         _one_hot_scores("TAB_2_3"), ["TAB_2_5"])
    assert parts["position"] == pytest.approx(2.0)
    # Split mass across same-string candidates averages the expectation.
    scores = torch.full((1, 1, len(TOKENS)), -20.0)
    scores[0, 0, IDS["TAB_2_3"]] = 20.0
    scores[0, 0, IDS["TAB_2_5"]] = 20.0
    _, parts = objective(LossConfig(position_alpha=1.0), scores, ["TAB_2_5"])
    assert parts["position"] == pytest.approx(1.0)
    # Open-string targets are excluded, like the reference formulation.
    _, parts = objective(LossConfig(position_alpha=1.0),
                         _one_hot_scores("TAB_6_2"), ["TAB_1_0"])
    assert parts["position"] == 0.0


def test_span_hinge_counts_onset_groups_and_margin():
    # Same onset, frets 9 and 5: span 4 == margin -> zero.
    scores = torch.cat([_one_hot_scores("TAB_3_9"), _one_hot_scores("TAB_2_5")], dim=1)
    _, parts = objective(LossConfig(span_alpha=1.0, span_margin=4), scores,
                         ["TAB_3_9", "TAB_2_5"])
    assert parts["span_groups"] == 1
    assert parts["span"] == 0.0
    # Expected frets 9 vs 3 across the onset: span 6, hinge at 4 -> 2.
    scores = torch.cat([_one_hot_scores("TAB_3_9"), _one_hot_scores("TAB_2_3")], dim=1)
    _, parts = objective(LossConfig(span_alpha=1.0, span_margin=4), scores,
                         ["TAB_3_9", "TAB_2_5"])
    assert parts["span"] == pytest.approx(2.0)
    # Single-note onsets and open strings never form a span group.
    scores = torch.cat([_one_hot_scores("TAB_3_9"),
                        torch.full((1, 1, len(TOKENS)), -20.0)], dim=1)
    scores[0, 1, IDS["TS_480"]] = 20.0
    _, parts = objective(LossConfig(span_alpha=1.0), scores,
                         ["TAB_3_9", "TS_480"])
    assert parts["span_groups"] == 0
    assert parts["span"] == 0.0


def test_stretch_hinge_tracks_local_smoothness():
    line = ["TAB_1_0", "TS_480", "TAB_1_2", "TS_480", "TAB_2_5",
            "TS_480", "TAB_1_2", "TS_480", "TAB_1_0"]
    scores = torch.cat([
        _one_hot_scores(t) if not t.startswith("TS_") else
        torch.full((1, 1, len(TOKENS)), -20.0).scatter_(2, torch.tensor([[[IDS[t]]]]), 20.0)
        for t in line], dim=1)
    _, parts = objective(LossConfig(stretch_alpha=1.0, stretch_margin=5),
                         scores, line)
    assert parts["stretch"] == 0.0
    # Same line with the middle note jumped an octave up the same pitch set
    # is impossible here without a new token, so force mass far away instead:
    # mass on TAB_6_2 (fret 2) for the middle TAB_2_5 keeps it smooth;
    # relocate mass to TAB_3_9's string is already tested; use margin 1.
    _, parts = objective(LossConfig(stretch_alpha=1.0, stretch_margin=1),
                         scores, line)
    # fretted targets [2,5,2] (opens excluded from the window means):
    # middle |5 - 3| = 2 -> hinge(2-1) = 1, sides |2-3| = 1 -> 0.
    assert parts["stretch"] == pytest.approx(1 / 3)


def test_playability_terms_disabled_by_default_and_combine():
    scores = torch.randn(1, 3, len(TOKENS), requires_grad=True)
    _, parts = objective(LossConfig(), scores, ["TAB_3_9", "TAB_2_5", "TS_480"])
    assert parts["position"] == 0.0 and parts["span"] == 0.0
    assert parts["stretch"] == 0.0 and parts["span_groups"] == 0
    config = LossConfig(position_alpha=2.0, span_alpha=3.0, stretch_alpha=0.5)
    loss, parts = objective(config, scores, ["TAB_3_9", "TS_480", "TAB_2_3"])
    assert torch.allclose(
        loss, parts["ce"] + 2.0 * parts["position"]
        + 3.0 * parts["span"] + 0.5 * parts["stretch"])
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_playability_terms_survive_padding_and_second_rows():
    scores = torch.randn(2, 3, len(TOKENS), requires_grad=True)
    labels = torch.tensor([[IDS["TAB_3_9"], IDS["TAB_2_5"], -100],
                           [IDS["TAB_1_2"], IDS["TS_480"], IDS["TAB_6_2"]]])
    config = LossConfig(position_alpha=1.0, span_alpha=1.0, stretch_alpha=1.0)
    loss, parts = TabLoss(META, config)(scores, labels, torch.cat([prefix(), prefix()]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(scores.grad).all()
    assert float(parts["span_groups"]) >= 0


def test_source_conditioning_handles_unseen_tuning_prefix_without_guessing():
    from open_fret.token_metadata import MAX_STRINGS
    scores = torch.randn(1, 1, len(TOKENS))
    labels = torch.tensor([[IDS["TAB_1_0"]]])
    module = TabLoss(META, LossConfig(ce_weight=0.3, pitch_alpha=1, smoothing=0.1))
    expected, _ = module(scores, labels, prefix())
    tuning = torch.tensor([[64, 59, 55, 50, 45, 40] + [-1] * (MAX_STRINGS - 6)])
    actual, _ = module(scores, labels, torch.tensor([[IDS["CAPO_0"], IDS["[UNK]"]]]),
                       loss_tuning=tuning, loss_capo=torch.tensor([0]))
    assert torch.allclose(actual, expected)
    with pytest.raises(ValueError, match="capo disagrees"):
        module(scores, labels, prefix(), loss_tuning=tuning, loss_capo=torch.tensor([2]))
