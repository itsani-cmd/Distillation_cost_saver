"""T5 attention with relative position bias and optional LoRA projections."""
import math
import numpy as np
import needle as ndl
from needle import nn, ops
from .lora import FrozenLinear, LoRALinear


def relative_position_buckets(query_length, key_length, bidirectional=True,
                              num_buckets=32, max_distance=128):
    """Implement T5's log-spaced relative position buckets in NumPy (indices only)."""
    relative = np.arange(key_length)[None, :] - np.arange(query_length)[:, None]
    if bidirectional:
        half = num_buckets // 2
        sign = (relative > 0).astype(np.int64) * half
        distance = np.abs(relative)
        buckets = half
    else:
        sign = 0
        distance = np.maximum(-relative, 0)
        buckets = num_buckets
    exact = buckets // 2
    distance_float = np.maximum(distance, 1)
    large = exact + (
        np.log(distance_float / exact) / math.log(max_distance / exact)
        * (buckets - exact)
    ).astype(np.int64)
    large = np.minimum(large, buckets - 1)
    return (sign + np.where(distance < exact, distance, large)).astype(np.float32)


def softmax_last(x):
    """Numerically stable softmax over the last axis using Needle operations."""
    z = ops.logsumexp(x, axes=(-1,))
    return ops.exp(x - z.reshape(x.shape[:-1] + (1,)).broadcast_to(x.shape))


class T5Attention(nn.Module):
    """One T5 attention module; weights use Needle's (in, out) convention."""

    def __init__(self, d_model, num_heads, d_kv, relative_bias=False,
                 bidirectional=False, rank=8, alpha=16, lora_dropout=0.0,
                 dropout=0.0, device=None, dtype="float32"):
        super().__init__()
        self.num_heads, self.d_kv = num_heads, d_kv
        self.inner_dim = num_heads * d_kv
        self.device, self.dtype = device, dtype
        self.bidirectional = bidirectional
        self.q = LoRALinear(d_model, self.inner_dim, rank, alpha, lora_dropout, device, dtype)
        self.k = LoRALinear(d_model, self.inner_dim, rank, alpha, lora_dropout, device, dtype)
        self.v = LoRALinear(d_model, self.inner_dim, rank, alpha, lora_dropout, device, dtype)
        self.o = LoRALinear(self.inner_dim, d_model, rank, alpha, lora_dropout, device, dtype)
        self.dropout = nn.Dropout(dropout)
        self.relative_attention_bias = (
            nn.Embedding(32, num_heads, device=device, dtype=dtype) if relative_bias else None
        )

    def split(self, x):
        batch, length, _ = x.shape
        return x.reshape((batch, length, self.num_heads, self.d_kv)).transpose((1, 2))

    def merge(self, x):
        batch, heads, length, d_kv = x.shape
        return x.transpose((1, 2)).reshape((batch, length, heads * d_kv))

    def position_bias(self, q_len, k_len):
        if self.relative_attention_bias is None:
            return ndl.Tensor(np.zeros((1, self.num_heads, q_len, k_len), dtype="float32"),
                              device=self.device, dtype=self.dtype, requires_grad=False)
        ids = relative_position_buckets(q_len, k_len, bidirectional=self.bidirectional)
        ids = ndl.Tensor(ids, device=self.device, dtype=self.dtype, requires_grad=False)
        return self.relative_attention_bias(ids).transpose((1, 2)).transpose((0, 1)).reshape(
            (1, self.num_heads, q_len, k_len)
        )

    def forward(self, hidden, key_value=None, mask=None, position_bias=None):
        source = hidden if key_value is None else key_value
        q, k, v = self.split(self.q(hidden)), self.split(self.k(source)), self.split(self.v(source))
        # T5 does NOT divide QK^T by sqrt(d_kv).
        scores = q @ k.transpose((2, 3))
        if position_bias is None:
            position_bias = self.position_bias(q.shape[2], k.shape[2])
            if mask is not None:
                position_bias = position_bias.broadcast_to(scores.shape) + mask.broadcast_to(scores.shape)
        scores = scores + position_bias.broadcast_to(scores.shape)
        weights = self.dropout(softmax_last(scores))
        return self.o(self.merge(weights @ v)), position_bias
