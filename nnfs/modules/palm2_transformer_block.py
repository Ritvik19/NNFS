import torch
import torch.nn as nn

from nnfs.layers import GroupedQueryAttention, RMSNorm, SwiGLUMLP


class PaLM2TransformerBlock(nn.Module):
    """Parallel Transformer Block as used in PaLM 2 (Anil et al., 2023).

    Uses a parallel formulation with pre-RMSNorm:
        y = x + Attention(RMSNorm(x)) + SwiGLUMLP(RMSNorm(x))
    where a single RMSNorm layer is shared prior to parallel sub-layer computation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.rms = RMSNorm(d_model, eps=eps)
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            use_rope=True,
            max_position_embeddings=max_position_embeddings,
            bias=False,
        )
        self.mlp = SwiGLUMLP(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normed_input = self.rms(input)
        attn_out = self.attn(normed_input)
        mlp_out = self.mlp(normed_input)
        return input + attn_out + mlp_out
