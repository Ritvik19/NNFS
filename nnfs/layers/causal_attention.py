import math

import torch
import torch.nn as nn

from .dropout import Dropout
from .linear import Linear


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_scale = 1 / math.sqrt(self.d_head)

        self.qkv = Linear(d_model, 3 * d_model)
        self.out = Linear(d_model, d_model)
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, C = input.shape
        qkv = self.qkv(input)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # (B, T, C) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.attn_scale
        causal_mask = torch.tril(torch.ones(T, T, device=input.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out(out))
