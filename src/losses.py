"""Training losses: label-smoothed CE and word-level knowledge distillation.

The combined objective is::

    loss = (1 - kd_weight) * CE + kd_weight * KD

where CE is label-smoothed cross-entropy against the gold labels and KD is a
temperature-scaled KL divergence between the student and teacher soft logits,
masked to valid (non-pad) target positions.

``kd_mode="sequence"`` is a documented extension point for Part-1+; it raises
``NotImplementedError`` so callers fail loudly rather than silently doing the
wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.data import IGNORE_INDEX


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Label-smoothed cross-entropy over target tokens.

    Args:
        logits: Student logits ``(batch, seq, vocab)``.
        labels: Gold token ids ``(batch, seq)`` with :data:`IGNORE_INDEX` on pads.
        label_smoothing: Smoothing factor in ``[0, 1)``.

    Returns:
        A scalar loss tensor.
    """
    vocab = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        label_smoothing=label_smoothing,
    )


def word_level_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Temperature-scaled word-level KL divergence KD loss.

    Computes ``KL(teacher || student)`` over the softmax distributions at each
    valid target position, scaled by ``temperature**2`` (the standard Hinton
    distillation correction so KD and CE gradients stay comparable in scale).

    Args:
        student_logits: ``(batch, seq, vocab)`` student logits (grad-tracked).
        teacher_logits: ``(batch, seq, vocab)`` teacher logits (no grad).
        labels: ``(batch, seq)`` labels; positions equal to :data:`IGNORE_INDEX`
            are excluded from the loss.
        temperature: Softmax temperature ``> 0``.

    Returns:
        A scalar loss tensor. Returns ``0`` if there are no valid positions.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    mask = labels != IGNORE_INDEX  # (batch, seq)
    if mask.sum() == 0:
        return student_logits.sum() * 0.0

    vocab = student_logits.size(-1)
    flat_mask = mask.reshape(-1)  # (batch*seq,)

    s_flat = student_logits.reshape(-1, vocab)[flat_mask]
    t_flat = teacher_logits.reshape(-1, vocab)[flat_mask]

    student_log_probs = F.log_softmax(s_flat / temperature, dim=-1)
    teacher_probs = F.softmax(t_flat / temperature, dim=-1)

    # batchmean divides by the number of (valid) rows -> mean per token.
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (temperature**2)


@dataclass
class LossOutput:
    """Breakdown of the combined loss for logging.

    Attributes:
        total: The combined scalar loss used for backprop.
        ce: The cross-entropy component.
        kd: The KD component (``0`` when distillation is off).
    """

    total: torch.Tensor
    ce: torch.Tensor
    kd: torch.Tensor


def combined_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    kd_weight: float = 0.0,
    temperature: float = 2.0,
    label_smoothing: float = 0.0,
    kd_mode: str = "word",
) -> LossOutput:
    """Compute ``(1 - kd_weight) * CE + kd_weight * KD``.

    When ``teacher_logits`` is ``None`` or ``kd_weight == 0`` this reduces to
    pure cross-entropy, so the same call site works for SFT and distillation.

    Args:
        student_logits: ``(batch, seq, vocab)`` student logits.
        labels: ``(batch, seq)`` gold labels with :data:`IGNORE_INDEX` on pads.
        teacher_logits: Optional teacher logits for KD.
        kd_weight: Weight on the KD term in ``[0, 1]``.
        temperature: KD softmax temperature.
        label_smoothing: CE label smoothing.
        kd_mode: ``"word"`` (implemented) or ``"sequence"`` (extension point).

    Returns:
        A :class:`LossOutput` with ``total``, ``ce``, and ``kd`` components.

    Raises:
        NotImplementedError: If ``kd_mode == "sequence"`` with KD active.
    """
    ce = cross_entropy_loss(student_logits, labels, label_smoothing=label_smoothing)

    if teacher_logits is None or kd_weight == 0.0:
        return LossOutput(total=ce, ce=ce, kd=torch.zeros((), device=ce.device))

    if kd_mode == "word":
        kd = word_level_kd_loss(student_logits, teacher_logits, labels, temperature=temperature)
    elif kd_mode == "sequence":
        # Extension point for Part 1+: sequence-level KD (e.g. distilling from
        # the teacher's generated sequences / top-k beams). Not implemented yet.
        raise NotImplementedError(
            "sequence-level KD is a documented extension point and is not "
            "implemented in Part 1; use kd_mode='word'."
        )
    else:
        raise ValueError(f"Unknown kd_mode: {kd_mode!r}")

    total = (1.0 - kd_weight) * ce + kd_weight * kd
    return LossOutput(total=total, ce=ce, kd=kd)
