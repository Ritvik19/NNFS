import torch
import torch.nn as nn

from nnfs.activations import GELU
from nnfs.layers import CausalMultiHeadAttention, LayerNorm, MLP


class GPT1TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = MLP(d_model, d_ff, GELU(), dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.ln1(input + self.attn(input))
        x = self.ln2(x + self.ffn(x))
        return x
