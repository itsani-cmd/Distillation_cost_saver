"""Masked translation cross-entropy and frozen-teacher KL, computed in Needle."""
import numpy as np
import needle as ndl
from needle import ops


def _masked_average(token_losses, valid):
    # valid is a constant Needle tensor; it has no gradient.
    denominator = float(np.sum(valid.numpy()))
    if denominator == 0:
        raise ValueError("Batch contains no valid target tokens")
    return ops.summation(token_losses * valid) / denominator


def masked_cross_entropy(logits, labels, valid_mask, device=None, dtype="float32"):
    """Label NLL using one-hot ground truth; padding labels must be safe token IDs."""
    batch, length, vocab = logits.shape
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype="float32")
    # Dense one-hot is simple but can be memory heavy for large T5 vocabulary.
    one_hot = np.zeros((batch * length, vocab), dtype="float32")
    one_hot[np.arange(batch * length), labels.reshape(-1)] = 1.0
    one_hot = one_hot.reshape(batch, length, vocab)
    target = ndl.Tensor(one_hot, device=device, dtype=dtype, requires_grad=False)
    mask = ndl.Tensor(valid, device=device, dtype=dtype, requires_grad=False)
    logsum = ops.logsumexp(logits, axes=(-1,))
    selected = ops.summation(logits * target, axes=(-1,))
    return _masked_average(logsum - selected, mask)


def teacher_kl(student_logits, teacher_logits, valid_mask, temperature=2.0,
               device=None, dtype="float32"):
    """KL(teacher || student), teacher distribution detached outside Needle."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher and student must share vocabulary and target positions")
    from .attention import softmax_last
    s = student_logits / temperature
    teacher = ndl.Tensor(np.asarray(teacher_logits, dtype="float32"),
                         device=device, dtype=dtype, requires_grad=False) / temperature
    p = softmax_last(teacher)
    ls = s - ops.logsumexp(s, axes=(-1,)).reshape(s.shape[:-1] + (1,)).broadcast_to(s.shape)
    lt = teacher - ops.logsumexp(teacher, axes=(-1,)).reshape(teacher.shape[:-1] + (1,)).broadcast_to(teacher.shape)
    per_token = ops.summation(p * (lt - ls), axes=(-1,))
    valid = ndl.Tensor(np.asarray(valid_mask, dtype="float32"),
                       device=device, dtype=dtype, requires_grad=False)
    return _masked_average(per_token, valid) * temperature**2


def distillation_loss(exit_logits, labels, valid_mask, teacher_logits,
                      kd_weight=0.5, temperature=2.0, device=None, dtype="float32"):
    """Apply CE + external KD at all exits, weighted toward deeper decoder layers."""
    if not (0 <= kd_weight <= 1):
        raise ValueError("kd_weight must be in [0, 1]")
    n = len(exit_logits)
    weights = [2 * (i + 1) / (n * (n + 1)) for i in range(n)]
    losses = []
    for logits in exit_logits:
        ce = masked_cross_entropy(logits, labels, valid_mask, device, dtype)
        kd = teacher_kl(logits, teacher_logits, valid_mask, temperature, device, dtype)
        losses.append((1 - kd_weight) * ce + kd_weight * kd)
    total = losses[0] * weights[0]
    for loss, weight in zip(losses[1:], weights[1:]):
        total = total + loss * weight
    return total
