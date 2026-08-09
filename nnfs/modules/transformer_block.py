import torch
import torch.nn as nn

from nnfs.activations import GELU, ReLU, SwiGLU
from nnfs.layers import CausalMultiHeadAttention, LayerNorm, MLP, SwiGLUMLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
        activation: str | nn.Module = "relu",
        use_rope: bool = False,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.attn = CausalMultiHeadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_position_embeddings=max_position_embeddings,
        )

        if isinstance(activation, str):
            act_str = activation.lower()
            if act_str == "relu":
                self.ffn = MLP(d_model, d_ff, ReLU(), dropout)
            elif act_str == "gelu":
                self.ffn = MLP(d_model, d_ff, GELU(), dropout)
            elif act_str == "swiglu":
                self.ffn = SwiGLUMLP(d_model, d_ff, dropout=dropout)
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
        elif isinstance(activation, (SwiGLU, SwiGLUMLP)):
            self.ffn = SwiGLUMLP(d_model, d_ff, dropout=dropout)
        elif isinstance(activation, nn.Module):
            self.ffn = MLP(d_model, d_ff, activation, dropout)
        else:
            raise ValueError(f"Invalid activation type: {type(activation)}")

        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, input: torch.Tensor, alibi_bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            x = input + self.attn(self.ln1(input), alibi_bias=alibi_bias)
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(input + self.attn(input, alibi_bias=alibi_bias))
            x = self.ln2(x + self.ffn(x))
        return x
