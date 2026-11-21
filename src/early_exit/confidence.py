"""Per-layer confidence measures (how sure is a layer about its word?).

Higher score = more confident = more willing to exit. All scores are in ``[0, 1]``.

Milestone 1 only needs :func:`softmax_gap` (used by the per-layer report).
The remaining measures from the CALM paper (hidden-state saturation, learned
classifier) are added in Part 2.
"""

from __future__ import annotations

import torch


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
