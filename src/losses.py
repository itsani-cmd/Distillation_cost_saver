"""Training losses: label-smoothed CE and word-level knowledge distillation.

The combined objective is::

    loss = (1 - kd_weight) * CE + kd_weight * KD

where CE is label-smoothed cross-entropy against the gold labels and KD is a
temperature-scaled KL divergence between the student and teacher soft logits,
masked to valid (non-pad) target positions.

``kd_mode="sequence"`` is a documented extension point for Part-1+; it raises
``NotImplementedError`` so callers fail loudly rather than silently doing the
wrong thing.

Early exit (Part 3): :func:`multi_exit_loss` applies the objective above to
*every* decoder layer's predictions (weighted by :func:`layer_weights`), plus
an optional self-distillation term pulling each early layer toward the final
layer's prediction.
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
        ce: The cross-entropy component. For the multi-exit loss this is the
            *final layer's* CE, so it stays comparable to non-early-exit runs.
        kd: The KD component (``0`` when distillation is off). Final layer's,
            for the multi-exit loss.
        per_layer_ce: Multi-exit only: detached CE of each layer, layer 1 first.
        self_distill: Multi-exit only: detached weighted self-distillation term.
    """

    total: torch.Tensor
    ce: torch.Tensor
    kd: torch.Tensor
    per_layer_ce: tuple[torch.Tensor, ...] = ()
    self_distill: torch.Tensor | None = None


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


def layer_weights(num_layers: int, scheme: str = "linear") -> list[float]:
    """Per-layer loss weights that sum to 1.

    ``"linear"`` gives layer ``i`` weight ``i / (1 + 2 + ... + L)`` (CALM paper,
    Eq. 6): later layers count more, so the final translation quality is
    mostly preserved. ``"uniform"`` gives every layer ``1 / L``.

    Args:
        num_layers: Number of exits ``L`` (``>= 1``).
        scheme: ``"linear"`` or ``"uniform"``.

    Returns:
        A list of ``L`` floats, layer 1 first.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if scheme == "linear":
        denom = num_layers * (num_layers + 1) / 2
        return [i / denom for i in range(1, num_layers + 1)]
    if scheme == "uniform":
        return [1.0 / num_layers] * num_layers
    raise ValueError(f"Unknown layer weighting scheme: {scheme!r}")


def multi_exit_loss(
    per_layer_logits: list[torch.Tensor],
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    kd_weight: float = 0.0,
    temperature: float = 2.0,
    label_smoothing: float = 0.0,
    kd_mode: str = "word",
    weighting: str = "linear",
    self_distill_weight: float = 0.0,
    self_distill_temperature: float = 1.0,
) -> LossOutput:
    """Grade every decoder layer's predictions (early-exit training).

    For each layer ``i`` (``L`` = last layer)::

        layer_loss_i = combined_loss(layer_i)                     # CE (+ teacher KD)
                     + self_distill_weight * KL(layer_L || layer_i)   # only for i < L
        total        = sum_i w_i * layer_loss_i

    The self-distillation target ``layer_L`` is **detached**: early layers are
    pulled toward the final layer, but the final layer is never pulled toward
    the (weaker) early layers.

    With a single layer and ``self_distill_weight == 0`` this is exactly
    :func:`combined_loss`.

    Args:
        per_layer_logits: ``L`` tensors ``(batch, seq, vocab)``, layer 1 first.
        labels: ``(batch, seq)`` gold labels with :data:`IGNORE_INDEX` on pads.
        teacher_logits: Optional teacher logits; teacher KD is applied to every layer.
        kd_weight: Teacher KD weight (see :func:`combined_loss`).
        temperature: Teacher KD temperature.
        label_smoothing: CE label smoothing.
        kd_mode: Teacher KD mode (see :func:`combined_loss`).
        weighting: Layer weighting scheme (see :func:`layer_weights`).
        self_distill_weight: Weight of the self-distillation term (``0`` = off).
        self_distill_temperature: Temperature of the self-distillation KL.

    Returns:
        A :class:`LossOutput`; ``ce``/``kd`` are the final layer's, and
        ``per_layer_ce`` / ``self_distill`` are filled in for logging.
    """
    if not per_layer_logits:
        raise ValueError("per_layer_logits must contain at least one layer")
    if self_distill_weight < 0:
        raise ValueError(f"self_distill_weight must be >= 0, got {self_distill_weight}")

    # Losses in float32: half-precision log-softmax over 32k words can hit -inf.
    per_layer_logits = [lg.float() for lg in per_layer_logits]
    weights = layer_weights(len(per_layer_logits), weighting)
    final_target = per_layer_logits[-1].detach()
    last = len(per_layer_logits) - 1

    total = None
    sd_total = torch.zeros((), device=labels.device)
    per_layer_ce: list[torch.Tensor] = []
    final_out: LossOutput | None = None

    for i, (w, logits) in enumerate(zip(weights, per_layer_logits)):
        out = combined_loss(
            student_logits=logits,
            labels=labels,
            teacher_logits=teacher_logits,
            kd_weight=kd_weight,
            temperature=temperature,
            label_smoothing=label_smoothing,
            kd_mode=kd_mode,
        )
        layer_loss = out.total
        if i < last and self_distill_weight > 0.0:
            sd = word_level_kd_loss(
                logits, final_target, labels, temperature=self_distill_temperature
            )
            layer_loss = layer_loss + self_distill_weight * sd
            sd_total = sd_total + w * sd.detach()
        total = w * layer_loss if total is None else total + w * layer_loss
        per_layer_ce.append(out.ce.detach())
        if i == last:
            final_out = out

    assert final_out is not None and total is not None
    return LossOutput(
        total=total,
        ce=final_out.ce,
        kd=final_out.kd,
        per_layer_ce=tuple(per_layer_ce),
        self_distill=sd_total,
    )
