"""Stopping rules: after each decoder layer, is the model sure enough to stop?

Three rules (Part 2):

* ``"softmax"`` -- CALM's *softmax response*: stop at the first layer whose
  top-1 minus top-2 word probability (:func:`softmax_gap`) reaches the threshold.
  This is the real method. An optional ``min_layer`` forbids stopping before a
  given layer (our layer 1 is often confidently wrong, see the Part 2 results).
* ``"static"`` -- always stop at a fixed layer ``k``. The "dumb" baseline every
  smart rule must beat.
* ``"oracle"`` -- stop at the first layer whose top word equals the LAST
  layer's top word. It cheats (needs all layers), so it is only an upper bound
  on what any rule can save -- and a check that copying states into skipped
  layers does not hurt quality.

:func:`pick_exit_layer` applies a rule to one word. Layers are 1-based
(layer 1 = first decoder layer, layer ``L`` = last).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

STOPPING_RULES = ("softmax", "static", "oracle")


def softmax_gap(logits: torch.Tensor) -> torch.Tensor:
    """CALM's "softmax response": top-1 probability minus top-2 probability.

    Example: "einen" 0.90 vs "ein" 0.04 -> 0.86 (very sure);
    "einen" 0.40 vs "ein" 0.35 -> 0.05 (unsure).

    Args:
        logits: ``(..., vocab)`` word scores.

    Returns:
        ``(...)`` tensor of scores in ``[0, 1]``.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    top2 = probs.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def pick_exit_layer(
    get_logits: Callable[[int], torch.Tensor],
    num_layers: int,
    rule: str,
    threshold: float = 0.9,
    static_layer: int = 3,
    min_layer: int = 1,
) -> tuple[int, torch.Tensor]:
    """Decide which layer one word stops at, and return that layer's word scores.

    Layers are requested lazily and in order (1, 2, ...), so a rule only pays
    for the layers it actually looks at.

    Args:
        get_logits: ``layer -> (vocab,)`` word scores for that 1-based layer.
        num_layers: Number of decoder layers ``L``.
        rule: One of :data:`STOPPING_RULES`.
        threshold: ``"softmax"`` only. Stop when ``softmax_gap >= threshold``.
            ``threshold >= 1`` means "never stop early" (always use layer ``L``).
        static_layer: ``"static"`` only. The layer to always stop at (clipped
            to ``[1, L]``).
        min_layer: ``"softmax"`` only. Never stop before this layer (clipped to
            ``[1, L]``); layers below it are not even scored. Useful when the
            shallowest layers are confidently wrong.

    Returns:
        ``(exit_layer, logits)`` -- the 1-based layer chosen and its word scores.

    Raises:
        ValueError: On an unknown rule or a bad ``num_layers``.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    if rule == "static":
        layer = min(max(static_layer, 1), num_layers)
        return layer, get_logits(layer)

    if rule == "softmax":
        for layer in range(min(max(min_layer, 1), num_layers), num_layers + 1):
            logits = get_logits(layer)
            if layer == num_layers:
                return layer, logits
            if threshold < 1.0 and softmax_gap(logits).item() >= threshold:
                return layer, logits

    if rule == "oracle":
        final = get_logits(num_layers)
        target = int(final.argmax(dim=-1))
        for layer in range(1, num_layers):
            logits = get_logits(layer)
            if int(logits.argmax(dim=-1)) == target:
                return layer, logits
        return num_layers, final

    raise ValueError(f"Unknown stopping rule {rule!r}; expected one of {STOPPING_RULES}")
