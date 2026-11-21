"""Early-exit greedy translation (CALM-style), one sentence at a time.

For every new German word::

    for layer 1..L:
        read this layer's word guess (exit head)
        stopping rule says "sure enough"?  -> output the word, stop at this layer
    copy the stopping layer's state into the layers this word skipped

**Why the copy?** Later words attend back to every earlier word at every
layer. If word 3 stopped at layer 2, layers 3..L have no state for it, so we
copy its layer-2 state upward (CALM paper Sec. 3.2/3.3.1, after Elbayad et
al.). Each higher layer still applies its *own* key/value projections to the
copied state; copying the keys/values themselves was found to hurt badly.

How it is implemented (version 1, built for correctness, not speed):

* Each step reruns T5's own decoder over the whole prefix (no KV cache).
* A forward hook on every decoder block overwrites the outputs for the
  *earlier* positions with the stored states -- including the copied ones --
  so the next block sees exactly what an early-exit model would see. It also
  records the current position's state at every layer.
* The stopping rule then reads the recorded states layer by layer and picks
  the exit layer.

Because all layers are still computed, this measures *how many layers the
model would need* (``exit_layers``), not wall-clock speed. A cached version
that really skips work can come later.

Public surface:
    - :class:`EarlyExitOutput`
    - :func:`early_exit_greedy`
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.early_exit.confidence import STOPPING_RULES, pick_exit_layer
from src.early_exit.heads import exit_logits, unwrap_t5
from src.early_exit.thresholds import constant_threshold


@dataclass
class EarlyExitOutput:
    """Result of translating one sentence with early exit.

    Attributes:
        token_ids: Generated token ids (without the decoder start token;
            includes the end-of-sentence token if one was produced).
        exit_layers: The 1-based layer each generated token stopped at.
    """

    token_ids: list[int] = field(default_factory=list)
    exit_layers: list[int] = field(default_factory=list)

    @property
    def avg_layers(self) -> float:
        """Average number of decoder layers used per generated token."""
        return sum(self.exit_layers) / max(len(self.exit_layers), 1)


@torch.no_grad()
def early_exit_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    rule: str = "softmax",
    threshold: float = 0.9,
    static_layer: int = 3,
    max_new_tokens: int = 127,
    min_exit_layer: int = 1,
) -> EarlyExitOutput:
    """Translate ONE sentence greedily, stopping each word at an early layer.

    Args:
        model: The (possibly LoRA/accelerate-wrapped) T5 model.
        input_ids: ``(1, src_len)`` source token ids (batch size must be 1).
        attention_mask: ``(1, src_len)`` source mask (defaults to all ones).
        rule: Stopping rule, one of :data:`~src.early_exit.confidence.STOPPING_RULES`.
        threshold: Softmax-rule threshold ``λ`` (``1.0`` = never stop early).
        static_layer: Static-rule layer.
        max_new_tokens: Maximum number of tokens to generate.
        min_exit_layer: Softmax rule only: never stop before this layer.

    Returns:
        An :class:`EarlyExitOutput` with the tokens and per-token exit layers.

    Raises:
        ValueError: If the batch size is not 1 or the rule is unknown.
    """
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError(
            f"early_exit_greedy needs a batch of 1, got shape {tuple(input_ids.shape)}"
        )
    if rule not in STOPPING_RULES:
        raise ValueError(f"Unknown stopping rule {rule!r}; expected one of {STOPPING_RULES}")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    t5 = unwrap_t5(model)
    was_training = t5.training
    t5.eval()
    blocks = t5.decoder.block
    n_layers = len(blocks)
    device = input_ids.device

    encoder_states = t5.get_encoder()(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    ).last_hidden_state
    start_id = t5.config.decoder_start_token_id
    eos_id = t5.config.eos_token_id

    # stored[i][p]: state of block i (0-based) at earlier position p, as an
    # early-exit model would have it (real if the word ran block i, else copied).
    stored: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    current: dict[int, torch.Tensor] = {}  # block index -> state at current position
    position = [0]  # current (last) decoder position; a list so hooks can read it

    def make_hook(i: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            t = position[0]
            if t > 0:
                hidden = hidden.clone()
                hidden[0, :t] = torch.stack(stored[i])
            current[i] = hidden[0, t].clone()
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        return hook

    logits_cache: dict[int, torch.Tensor] = {}  # 1-based layer -> word scores, this step

    def get_logits(layer: int) -> torch.Tensor:
        """Word scores of the current word at ``layer`` (computed once per step)."""
        if layer not in logits_cache:
            state = current[layer - 1].view(1, 1, -1)
            logits_cache[layer] = exit_logits(t5, state)[0, 0]
        return logits_cache[layer]

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(blocks)]
    result = EarlyExitOutput()
    dec_ids = [start_id]
    try:
        for _ in range(max_new_tokens):
            position[0] = len(dec_ids) - 1
            current.clear()
            logits_cache.clear()
            t5.decoder(
                input_ids=torch.tensor([dec_ids], device=device),
                encoder_hidden_states=encoder_states,
                encoder_attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

            exit_layer, logits = pick_exit_layer(
                get_logits,
                n_layers,
                rule,
                threshold=constant_threshold(threshold, position[0]),
                static_layer=static_layer,
                min_layer=min_exit_layer,
            )
            token = int(logits.argmax(dim=-1))

            # Real states up to the exit layer; the exit state copied above it.
            for i in range(n_layers):
                stored[i].append(current[i] if i < exit_layer else current[exit_layer - 1])

            dec_ids.append(token)
            result.token_ids.append(token)
            result.exit_layers.append(exit_layer)
            if token == eos_id:
                break
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            t5.train()
    return result
