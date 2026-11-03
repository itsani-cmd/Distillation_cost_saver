"""Tests for CE and word-level KD losses."""

import torch

from src.data import IGNORE_INDEX
from src.losses import (
    combined_loss,
    cross_entropy_loss,
    word_level_kd_loss,
)


def _toy_logits(batch=2, seq=3, vocab=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, vocab, generator=g)


def test_kd_zero_when_logits_equal():
    logits = _toy_logits()
    labels = torch.zeros(2, 3, dtype=torch.long)
    kd = word_level_kd_loss(logits, logits.clone(), labels, temperature=2.0)
    assert torch.isclose(kd, torch.tensor(0.0), atol=1e-5)


def test_kd_ignores_padding_positions():
    student = _toy_logits(seed=1)
    teacher = _toy_logits(seed=2)
    labels_full = torch.tensor([[1, 2, 3], [1, 2, 3]])
    # Mask out everything -> KD must be exactly zero.
    labels_masked = torch.full_like(labels_full, IGNORE_INDEX)
    kd = word_level_kd_loss(student, teacher, labels_masked, temperature=2.0)
    assert torch.isclose(kd, torch.tensor(0.0), atol=1e-7)


def test_combined_reduces_to_ce_when_kd_weight_zero():
    student = _toy_logits(seed=3)
    teacher = _toy_logits(seed=4)
    labels = torch.tensor([[1, 2, 3], [0, 1, IGNORE_INDEX]])

    ce = cross_entropy_loss(student, labels, label_smoothing=0.0)
    out = combined_loss(
        student_logits=student,
        labels=labels,
        teacher_logits=teacher,
        kd_weight=0.0,
        temperature=2.0,
    )
    assert torch.isclose(out.total, ce)
    assert torch.isclose(out.kd, torch.tensor(0.0))


def test_combined_mixes_ce_and_kd():
    student = _toy_logits(seed=5)
    teacher = _toy_logits(seed=6)
    labels = torch.tensor([[1, 2, 3], [0, 1, 2]])

    out = combined_loss(
        student_logits=student,
        labels=labels,
        teacher_logits=teacher,
        kd_weight=0.5,
        temperature=2.0,
    )
    expected = 0.5 * out.ce + 0.5 * out.kd
    assert torch.isclose(out.total, expected, atol=1e-6)
    assert out.kd.item() > 0.0


def test_no_teacher_is_pure_ce():
    student = _toy_logits(seed=7)
    labels = torch.tensor([[1, 2, 3], [0, 1, 2]])
    out = combined_loss(student_logits=student, labels=labels, teacher_logits=None)
    assert torch.isclose(out.total, out.ce)
