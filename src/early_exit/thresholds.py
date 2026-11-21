"""Exit thresholds for the softmax stopping rule.

Only a **constant** threshold is implemented: the same ``λ`` for every word in
the sentence. Sweeping ``λ`` from strict (0.9) to loose (0.1) gives the
speed-vs-quality table.

The CALM paper's *decaying* threshold (strict early in the sentence, looser
later) is a possible later add-on; it mainly helps on long outputs.
"""

from __future__ import annotations


def constant_threshold(threshold: float, position: int = 0) -> float:
    """Threshold for the word at ``position``: the same value everywhere.

    Args:
        threshold: The exit threshold ``λ`` in ``[0, 1]`` (``1`` = never exit early).
        position: 0-based position of the word being generated. Unused by the
            constant threshold; kept so a decaying threshold can share the signature.

    Returns:
        ``threshold``.

    Raises:
        ValueError: If ``threshold`` is outside ``[0, 1]``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return threshold
