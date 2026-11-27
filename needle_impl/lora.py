"""LoRA projections using Needle tensors; base parameters stay out of the optimizer."""

import math
import numpy as np
import needle as ndl
from needle import nn


def linear_last_dim(x, weight):
    """Apply a 2D weight matrix to the last dimension of a tensor."""
    assert x.shape[-1] == weight.shape[0]

    input_shape = x.shape
    batch = math.prod(input_shape[:-1])

    # Flatten all leading dimensions before matrix multiplication.
    flat_x = x.reshape((batch, input_shape[-1]))
    flat_output = flat_x @ weight

    # Restore the original leading dimensions.
    output_shape = input_shape[:-1] + (weight.shape[1],)
    return flat_output.reshape(output_shape)


class LoRALinear(nn.Module):
    """A bias-free linear layer: xW + (alpha/r) * dropout(x)AB."""

    def __init__(self, input_dim, output_dim, rank=8, alpha=16,
                 dropout=0.0, device=None, dtype="float32"):
        super().__init__()

        if rank < 1:
            raise ValueError("LoRA rank must be positive")

        self.weight = nn.Parameter(ndl.Tensor(
            np.zeros((input_dim, output_dim), dtype=np.float32),
            device=device, dtype=dtype
        ))

        self.A = nn.Parameter(ndl.Tensor(
            np.random.normal(
                0, 1 / rank, (input_dim, rank)
            ).astype("float32"),
            device=device, dtype=dtype
        ))

        self.B = nn.Parameter(ndl.Tensor(
            np.zeros((rank, output_dim), dtype=np.float32),
            device=device, dtype=dtype
        ))

        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        # Base projection
        base = linear_last_dim(x, self.weight)

        # Low-rank adapter projection
        adapter = linear_last_dim(self.dropout(x), self.A)
        adapter = linear_last_dim(adapter, self.B)

        return base + adapter * self.scale

    def trainable_parameters(self):
        return [self.A, self.B]


class FrozenLinear(nn.Module):
    """A bias-free projection with a frozen pretrained weight."""

    def __init__(self, input_dim, output_dim, device=None, dtype="float32"):
        super().__init__()

        self.weight = nn.Parameter(ndl.Tensor(
            np.zeros((input_dim, output_dim), dtype=np.float32),
            device=device, dtype=dtype
        ))

    def forward(self, x):
        return linear_last_dim(x, self.weight)

    def trainable_parameters(self):
        return []
