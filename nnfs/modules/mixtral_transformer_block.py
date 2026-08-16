import torch
import torch.nn as nn

from nnfs.layers import GroupedQueryAttention, RMSNorm, SparseMoE


class MixtralTransformerBlock(nn.Module):
    """Mixtral Transformer Block.

    Features:
    - Pre-normalization using RMSNorm
    - Grouped-Query Attention (GQA) with RoPE and optional Sliding Window Attention (SWA)
    - Sparse Mixture-of-Experts (MoE) sub-layer (8 SwiGLU experts, Top-2 router)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        d_ff: int | None = None,
        num_experts: int = 8,
        top_k_experts: int = 2,
        dropout: float = 0.0,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
        sliding_window: int | None = 4096,
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
            sliding_window=sliding_window,
        )
        self.rms_2 = RMSNorm(d_model, eps=eps)
        self.moe = SparseMoE(
            d_model=d_model,
            d_ff=d_ff if d_ff is not None else 1024,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            dropout=dropout,
            bias=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input + self.attn(self.rms_1(input))
        x = x + self.moe(self.rms_2(x))
        return x
