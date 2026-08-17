import torch
import torch.nn as nn

from nnfs.layers import Llama4Attention, RMSNorm, SharedSparseMoE


class Llama4TransformerBlock(nn.Module):
    """Llama 4 Transformer Block.

    Features:
    - Pre-normalization using RMSNorm
    - Llama 4 Attention supporting iRoPE (chunked RoPE vs global NoPE layers) and attention temperature scaling
    - Shared-and-Routed Sparse Mixture-of-Experts (MoE) with 1 shared SwiGLU expert and Top-K routed SwiGLU experts
    - Bias-free linear projections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        d_head: int | None = None,
        d_ff: int | None = None,
        d_ff_shared: int | None = None,
        num_experts: int = 16,
        top_k_experts: int = 1,
        dropout: float = 0.0,
        is_rope_layer: bool = True,
        chunk_size: int | None = 8192,
        max_position_embeddings: int = 8192,
        rope_theta: float = 500000.0,
        rope_scaling: dict | None = None,
        temp_scaling: float = 1.0,
        clamp_limit: float | None = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.is_rope_layer = is_rope_layer

        self.rms_1 = RMSNorm(d_model, eps=eps)
        self.attn = Llama4Attention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            dropout=dropout,
            is_rope_layer=is_rope_layer,
            chunk_size=chunk_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            temp_scaling=temp_scaling,
            bias=False,
        )

        self.rms_2 = RMSNorm(d_model, eps=eps)
        self.moe = SharedSparseMoE(
            d_model=d_model,
            d_ff=d_ff if d_ff is not None else d_model * 4,
            d_ff_shared=d_ff_shared,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            dropout=dropout,
            bias=False,
            clamp_limit=clamp_limit,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input + self.attn(self.rms_1(input))
        x = x + self.moe(self.rms_2(x))
        return x
