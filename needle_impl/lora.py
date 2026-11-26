"""LoRA projections using Needle tensors; base parameters stay out of the optimizer."""
import numpy as np
import needle as ndl
from needle import nn


class LoRALinear(nn.Module):
    """A bias-free linear layer: xW + (alpha/r) * dropout(x)AB."""

    def __init__(self, input_dim, output_dim, rank=8, alpha=16, dropout=0.0,
                 device=None, dtype="float32"):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.weight = nn.Parameter(ndl.Tensor(
            np.zeros((input_dim, output_dim), dtype=np.float32), device=device, dtype=dtype
        ))
        self.A = nn.Parameter(ndl.Tensor(
            np.random.normal(0, 1 / rank, (input_dim, rank)).astype("float32"),
            device=device, dtype=dtype
        ))
        self.B = nn.Parameter(ndl.Tensor(
            np.zeros((rank, output_dim), dtype=np.float32), device=device, dtype=dtype
        ))
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        return (x @ self.weight) + ((self.dropout(x) @ self.A) @ self.B) * self.scale

    def trainable_parameters(self):
        return [self.A, self.B]


class FrozenLinear(nn.Module):
    """A bias-free projection with a frozen pretrained weight."""

    def __init__(self, input_dim, output_dim, device=None, dtype="float32"):
        super().__init__()
        self.weight = nn.Parameter(ndl.Tensor(
            np.zeros((input_dim, output_dim), dtype=np.float32), device=device, dtype=dtype
        ))

    def forward(self, x):
        return x @ self.weight

    def trainable_parameters(self):
        return []
