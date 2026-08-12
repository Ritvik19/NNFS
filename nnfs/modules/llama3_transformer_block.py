import torch
import torch.nn as nn

from nnfs.layers import GroupedQueryAttention, RMSNorm, SwiGLUMLP


class Llama3TransformerBlock(nn.Module):
    """LLaMA 3 Transformer Block.

    Features:
    - Pre-normalization using RMSNorm
    - Grouped-Query Attention (GQA) with Llama 3 RoPE (theta = 500k, optional piecewise scaling) and bias-free projections
    - SwiGLU Feed-Forward Network with bias-free projections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        d_ff: int | None = None,
        dropout: float = 0.0,
        max_position_embeddings: int = 8192,
        rope_theta: float = 500000.0,
        rope_scaling: dict | None = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.rms_1 = RMSNorm(d_model, eps=eps)
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            use_rope=True,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            bias=False,
        )
        self.rms_2 = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLUMLP(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input + self.attn(self.rms_1(input))
        x = x + self.ffn(self.rms_2(x))
        return x
