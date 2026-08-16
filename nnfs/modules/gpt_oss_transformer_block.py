import torch
import torch.nn as nn

from nnfs.layers import GptOssAttention, RMSNorm, SparseMoE


class GptOssTransformerBlock(nn.Module):
    """GPT-OSS Transformer Block.

    Features from OpenAI GPT-OSS (arXiv:2508.10925):
    - Pre-normalization using RMSNorm
    - GptOssAttention (GQA, learned attention sink denominator bias, alternating sliding window / full attention, RoPE/YaRN)
    - Sparse Mixture-of-Experts (MoE) with clamped SwiGLU activations and Top-K router
    - Residual connections around both sub-layers
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        d_head: int | None = None,
        d_ff: int | None = None,
        num_experts: int = 32,
        top_k_experts: int = 4,
        dropout: float = 0.0,
        max_position_embeddings: int = 131072,
        rope_theta: float = 150000.0,
        rope_scaling: dict | None = None,
        sliding_window: int | None = 128,
        swiglu_limit: float | None = 7.0,
        attention_bias: bool = True,
        sink_bias: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.rms_1 = RMSNorm(d_model, eps=eps)
        self.attn = GptOssAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            dropout=dropout,
            use_rope=True,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            bias=attention_bias,
            sliding_window=sliding_window,
            sink_bias=sink_bias,
        )
        self.rms_2 = RMSNorm(d_model, eps=eps)
        self.moe = SparseMoE(
            d_model=d_model,
            d_ff=d_ff if d_ff is not None else d_model,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            dropout=dropout,
            bias=False,
            clamp_limit=swiglu_limit,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input + self.attn(self.rms_1(input))
        x = x + self.moe(self.rms_2(x))
        return x
