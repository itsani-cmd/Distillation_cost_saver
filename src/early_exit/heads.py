"""Exit heads: turn any decoder layer's hidden state into word scores.

T5 normally only predicts a word after the last decoder layer::

    logits = lm_head( scale * dropout( final_layer_norm( h_L ) ) )

where ``scale = d_model ** -0.5`` when input/output embeddings are tied (true
for ``t5-small``). An *exit head* applies the exact same recipe to an earlier
layer's hidden state ``h_i``. Every layer shares the same ``final_layer_norm``
and ``lm_head`` -- no new parameters are added (as in the CALM paper).

HF quirk handled here: with ``output_hidden_states=True`` the decoder returns
``L + 1`` hidden states::

    decoder_hidden_states[0]      -> token embeddings (input to layer 1)
    decoder_hidden_states[1..L-1] -> raw outputs of layers 1..L-1 (NOT normed)
    decoder_hidden_states[L]      -> output of layer L, ALREADY normed + dropout

So layers ``1..L-1`` need the norm applied, but layer ``L`` must not get it twice.

Public surface:
    - :func:`unwrap_t5` -> the plain ``T5ForConditionalGeneration`` inside peft/accelerate wrappers
    - :func:`num_decoder_layers`
    - :func:`exit_logits` -> word scores for one intermediate layer's raw hidden state
    - :func:`all_exit_logits` -> word scores for every decoder layer
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def unwrap_t5(model: nn.Module) -> nn.Module:
    """Return the underlying HF T5 model from accelerate / peft wrappers.

    Args:
        model: A T5 model, possibly wrapped by DDP (``.module``) and/or peft
            (``get_base_model()``).

    Returns:
        The inner ``T5ForConditionalGeneration`` (LoRA layers stay injected in
        it, so calling its submodules still uses the adapter).
    """
    from peft import PeftModel

    inner = getattr(model, "module", model)  # DDP / accelerate wrapper
    if isinstance(inner, PeftModel):
        inner = inner.get_base_model()
    return inner


def num_decoder_layers(model: nn.Module) -> int:
    """Number of decoder layers (6 for ``t5-small``)."""
    return len(unwrap_t5(model).decoder.block)


def exit_logits(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Word scores from a *raw* (not yet normed) intermediate decoder hidden state.

    Args:
        model: The (possibly wrapped) T5 model.
        hidden: ``(batch, seq, d_model)`` output of some decoder layer ``i < L``,
            i.e. ``decoder_hidden_states[i]`` for ``1 <= i <= L - 1``.

    Returns:
        ``(batch, seq, vocab)`` logits, computed exactly like the final layer's.
    """
    t5 = unwrap_t5(model)
    h = t5.decoder.final_layer_norm(hidden)
    h = t5.decoder.dropout(h)
    return _project(t5, h)


def _project(t5: nn.Module, normed_hidden: torch.Tensor) -> torch.Tensor:
    """Apply T5's tied-embedding rescale and the shared ``lm_head``."""
    if getattr(t5.config, "tie_word_embeddings", True):
        normed_hidden = normed_hidden * (t5.config.d_model**-0.5)
    return t5.lm_head(normed_hidden)


def all_exit_logits(
    model: nn.Module,
    decoder_hidden_states: Sequence[torch.Tensor],
    final_logits: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Word scores for every decoder layer, layer 1 first.

    Args:
        model: The (possibly wrapped) T5 model.
        decoder_hidden_states: The ``L + 1`` hidden states HF returns with
            ``output_hidden_states=True`` (see the module docstring).
        final_logits: Optional, the model's own ``outputs.logits``. If given it
            is reused for the last layer instead of recomputing it.

    Returns:
        A list of ``L`` tensors of shape ``(batch, seq, vocab)``; entry ``i - 1``
        is layer ``i``'s prediction.
    """
    t5 = unwrap_t5(model)
    n_layers = len(t5.decoder.block)
    if len(decoder_hidden_states) != n_layers + 1:
        raise ValueError(
            f"expected {n_layers + 1} decoder hidden states, got {len(decoder_hidden_states)}"
        )
    per_layer = [exit_logits(t5, decoder_hidden_states[i]) for i in range(1, n_layers)]
    if final_logits is None:
        # Already normed + dropout by HF; only rescale + project.
        final_logits = _project(t5, decoder_hidden_states[n_layers])
    per_layer.append(final_logits)
    return per_layer
