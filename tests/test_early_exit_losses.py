"""Tests for the multi-exit (every-layer) loss and the softmax-gap confidence."""

import pytest
import torch

from src.data import IGNORE_INDEX
from src.early_exit.confidence import softmax_gap
from src.losses import (
    LossOutput,
    combined_loss,
    layer_weights,
    multi_exit_loss,
    word_level_kd_loss,
)


def _logits(seed, batch=2, seq=3, vocab=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, vocab, generator=g)


LABELS = torch.tensor([[1, 2, 3], [0, 4, IGNORE_INDEX]])


def test_linear_weights_match_paper():
    w = layer_weights(6, "linear")
    assert w == pytest.approx([i / 21 for i in range(1, 7)])
    assert sum(w) == pytest.approx(1.0)
    assert all(a < b for a, b in zip(w, w[1:]))  # later layers count more


def test_uniform_weights():
    assert layer_weights(4, "uniform") == pytest.approx([0.25] * 4)


def test_layer_weights_rejects_bad_input():
    with pytest.raises(ValueError):
        layer_weights(0)
    with pytest.raises(ValueError):
        layer_weights(3, "cubic")


def test_single_layer_equals_combined_loss():
    logits = _logits(0)
    out = multi_exit_loss([logits], LABELS, label_smoothing=0.1)
    ref = combined_loss(logits, LABELS, label_smoothing=0.1)
    assert torch.isclose(out.total, ref.total)
    assert torch.isclose(out.ce, ref.ce)
    assert len(out.per_layer_ce) == 1


def test_total_is_weighted_sum_with_self_distill():
    layers = [_logits(1), _logits(2), _logits(3)]
    sdw, temp = 0.5, 2.0
    out = multi_exit_loss(layers, LABELS, self_distill_weight=sdw, self_distill_temperature=temp)

    w = layer_weights(3, "linear")
    expected = torch.zeros(())
    expected_sd = torch.zeros(())
    for i, lg in enumerate(layers):
        term = combined_loss(lg, LABELS).total
        if i < 2:
            sd = word_level_kd_loss(lg, layers[-1], LABELS, temperature=temp)
            term = term + sdw * sd
            expected_sd = expected_sd + w[i] * sd
        expected = expected + w[i] * term

    assert torch.isclose(out.total, expected, atol=1e-6)
    assert torch.isclose(out.self_distill, expected_sd, atol=1e-6)
    # ce/kd report the FINAL layer, so logs stay comparable with old runs.
    assert torch.isclose(out.ce, combined_loss(layers[-1], LABELS).ce)
    assert len(out.per_layer_ce) == 3


def test_self_distill_zero_when_layers_agree():
    same = _logits(4)
    out = multi_exit_loss([same.clone(), same.clone()], LABELS, self_distill_weight=1.0)
    assert torch.isclose(out.self_distill, torch.tensor(0.0), atol=1e-6)


def test_self_distill_does_not_push_on_final_layer():
    """The final layer's gradient must be identical with and without self-distill."""

    def final_grad(sdw):
        early = _logits(5).requires_grad_(True)
        final = _logits(6).requires_grad_(True)
        multi_exit_loss([early, final], LABELS, self_distill_weight=sdw).total.backward()
        return final.grad.clone(), early.grad.clone()

    final_no_sd, early_no_sd = final_grad(0.0)
    final_sd, early_sd = final_grad(1.0)
    assert torch.allclose(final_no_sd, final_sd)
    # ...while the early layer's gradient DOES change.
    assert not torch.allclose(early_no_sd, early_sd)


def test_teacher_kd_applies_to_every_layer():
    layers = [_logits(7), _logits(8)]
    teacher = _logits(9)
    out = multi_exit_loss(layers, LABELS, teacher_logits=teacher, kd_weight=0.5)
    w = layer_weights(2)
    expected = sum(
        wi * combined_loss(lg, LABELS, teacher_logits=teacher, kd_weight=0.5).total
        for wi, lg in zip(w, layers)
    )
    assert torch.isclose(out.total, expected, atol=1e-6)


def test_multi_exit_rejects_bad_input():
    with pytest.raises(ValueError):
        multi_exit_loss([], LABELS)
    with pytest.raises(ValueError):
        multi_exit_loss([_logits(0)], LABELS, self_distill_weight=-1.0)


def test_loss_output_defaults_keep_old_call_sites_working():
    out = LossOutput(total=torch.tensor(1.0), ce=torch.tensor(1.0), kd=torch.tensor(0.0))
    assert out.per_layer_ce == ()
    assert out.self_distill is None


def test_softmax_gap():
    sure = torch.tensor([[10.0, 0.0, 0.0]])
    unsure = torch.tensor([[1.0, 1.0, 0.0]])
    assert softmax_gap(sure).item() > 0.99
    assert softmax_gap(unsure).item() == pytest.approx(0.0, abs=1e-6)
    gaps = softmax_gap(_logits(10))
    assert gaps.shape == (2, 3)
    assert ((gaps >= 0) & (gaps <= 1)).all()


def test_format_log_line_includes_layers():
    pytest.importorskip("accelerate")
    from src.train import format_log_line

    out = multi_exit_loss([_logits(11), _logits(12)], LABELS, self_distill_weight=0.5)
    line = format_log_line(0, 10, out)
    assert "ce_L1=" in line and "ce_L2=" in line and "sd=" in line
    plain = format_log_line(0, 10, combined_loss(_logits(11), LABELS))
    assert "ce_L1" not in plain and "sd=" not in plain
