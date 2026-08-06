import torch
import torch.nn as nn

from nnfs.layers import LayerNorm, MultiQueryAttention, SwiGLUMLP


class PaLMTransformerBlock(nn.Module):
    """Parallel Transformer Block as introduced in PaLM (Chowdhery et al., 2022).

    Uses a parallel formulation:
        y = x + MultiQueryAttention(LayerNorm(x)) + SwiGLUMLP(LayerNorm(x))
    where a single LayerNorm is shared prior to parallel computation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.ln = LayerNorm(d_model)
        self.attn = MultiQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=False,
            max_position_embeddings=max_position_embeddings,
        )
        self.mlp = SwiGLUMLP(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normed_input = self.ln(input)
        attn_out = self.attn(normed_input)
        mlp_out = self.mlp(normed_input)
        return input + attn_out + mlp_out
