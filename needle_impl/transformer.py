"""T5-small-compatible encoder and decoder blocks with Needle computation."""
import numpy as np
import needle as ndl
from needle import nn, ops
from .attention import T5Attention
from .lora import FrozenLinear


def attention_mask(valid_keys, query_length, causal=False, device=None, dtype="float32"):
    """Build a broadcastable additive mask, using NumPy for non-trainable control."""
    valid = np.asarray(valid_keys, dtype=bool)
    batch, key_length = valid.shape
    blocked = np.broadcast_to(~valid[:, None, :], (batch, query_length, key_length)).copy()
    if causal:
        blocked |= (np.arange(key_length)[None, :] > np.arange(query_length)[:, None])[None]
    return ndl.Tensor((blocked * -1e9)[:, None].astype("float32"),
                      device=device, dtype=dtype, requires_grad=False)


class T5LayerNorm(nn.Module):
    """T5 RMS-style norm: no mean subtraction and no additive bias."""

    def __init__(self, d_model, eps=1e-6, device=None, dtype="float32"):
        super().__init__()
        self.weight = nn.Parameter(ndl.Tensor(np.ones((d_model,), dtype="float32"),
                                               device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        d = x.shape[-1]
        variance = ops.summation(x * x, axes=(-1,)) / d
        scale = ops.power_scalar(variance + self.eps, -0.5)
        return x * scale.reshape(x.shape[:-1] + (1,)).broadcast_to(x.shape) * self.weight.broadcast_to(x.shape)


class T5FF(nn.Module):
    """The non-gated ReLU feed-forward block used by original t5-small."""

    def __init__(self, d_model, d_ff, dropout, device=None, dtype="float32"):
        super().__init__()
        self.wi = FrozenLinear(d_model, d_ff, device, dtype)
        self.wo = FrozenLinear(d_ff, d_model, device, dtype)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.wo(self.dropout(self.relu(self.wi(x))))


class T5Block(nn.Module):
    """T5 encoder block, or decoder block with cross-attention."""

    def __init__(self, config, decoder, first, device=None, dtype="float32"):
        super().__init__()
        d, heads, dk = config.d_model, config.num_heads, config.d_kv
        dropout = config.dropout_rate
        self.decoder = decoder
        self.self_norm = T5LayerNorm(d, config.layer_norm_epsilon, device, dtype)
        self.self_attn = T5Attention(d, heads, dk, first, not decoder,
                                     config.lora_rank, config.lora_alpha,
                                     config.lora_dropout, dropout, device, dtype)
        self.dropout = nn.Dropout(dropout)
        if decoder:
            self.cross_norm = T5LayerNorm(d, config.layer_norm_epsilon, device, dtype)
            self.cross_attn = T5Attention(d, heads, dk, False, False,
                                          config.lora_rank, config.lora_alpha,
                                          config.lora_dropout, dropout, device, dtype)
        self.ff_norm = T5LayerNorm(d, config.layer_norm_epsilon, device, dtype)
        self.ff = T5FF(d, config.d_ff, dropout, device, dtype)

    def forward(self, hidden, self_mask, encoder_hidden=None, cross_mask=None,
                self_bias=None, cross_bias=None):
        update, self_bias = self.self_attn(self.self_norm(hidden), mask=self_mask,
                                           position_bias=self_bias)
        hidden = hidden + self.dropout(update)
        if self.decoder:
            update, cross_bias = self.cross_attn(self.cross_norm(hidden), encoder_hidden,
                                                 mask=cross_mask, position_bias=cross_bias)
            hidden = hidden + self.dropout(update)
        hidden = hidden + self.dropout(self.ff(self.ff_norm(hidden)))
        return hidden, self_bias, cross_bias


class T5Stack(nn.Module):
    """Stack decoder or encoder blocks, sharing position bias across all layers."""

    def __init__(self, config, decoder, device=None, dtype="float32"):
        super().__init__()
        count = config.num_decoder_layers if decoder else config.num_layers
        self.layers = [T5Block(config, decoder, i == 0, device, dtype) for i in range(count)]
        self.final_layer_norm = T5LayerNorm(config.d_model, config.layer_norm_epsilon, device, dtype)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.decoder = decoder

    def forward(self, hidden, mask, encoder_hidden=None, cross_mask=None,
                return_all_exits=False):
        hidden = self.dropout(hidden)
        self_bias = cross_bias = None
        exits = []
        for layer in self.layers:
            hidden, self_bias, cross_bias = layer(hidden, mask, encoder_hidden, cross_mask,
                                                   self_bias, cross_bias)
            if return_all_exits:
                exits.append(self.final_layer_norm(hidden))
        hidden = self.dropout(self.final_layer_norm(hidden))
        if return_all_exits:
            exits[-1] = hidden  # HF final logits use final dropout after final norm.
            return hidden, exits
        return hidden
