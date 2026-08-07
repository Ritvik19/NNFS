import torch
import torch.nn as nn

from nnfs.activations import ReLU
from nnfs.layers import CausalMultiHeadAttention, LayerNorm, MLP


class VaswaniTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.attn = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = MLP(d_model, d_ff, ReLU(), dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = input + self.attn(self.ln1(input))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(input + self.attn(input))
            x = self.ln2(x + self.ffn(x))
        return x
