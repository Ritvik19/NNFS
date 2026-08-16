import math
import torch
import torch.nn as nn

from .dropout import Dropout
from .linear import Linear
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class GptOssAttention(nn.Module):
    """GPT-OSS Attention with Grouped-Query Attention (GQA), Learned Attention Sinks,
    Alternating Sliding-Window Attention, and Rotary Position Embeddings (RoPE/YaRN).

    Key features from OpenAI GPT-OSS (arXiv:2508.10925):
    - Learned attention sink bias in the softmax denominator, enabling heads to attend to 'null'.
    - Grouped-Query Attention (GQA) with optional projection biases.
    - Flexible query/head dimension decoupled from d_model (e.g. 64 heads * 64 dim = 4096).
    - Sliding-window causal attention or full dense causal attention per layer.
    - Rotary Positional Embeddings with YaRN scaling.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        d_head: int | None = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_position_embeddings: int = 4096,
        rope_theta: float = 150000.0,
        rope_scaling: dict | None = None,
        bias: bool = True,
        sliding_window: int | None = None,
        sink_bias: bool = True,
    ):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head if d_head is not None else (d_model // n_heads)

        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"

        self.n_rep = n_heads // n_kv_heads
        self.attn_scale = 1.0 / math.sqrt(self.d_head)
        self.use_rope = use_rope
        self.sliding_window = sliding_window

        self.q_proj = Linear(d_model, n_heads * self.d_head, bias=bias)
        self.k_proj = Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.v_proj = Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.out_proj = Linear(n_heads * self.d_head, d_model, bias=bias)

        if sink_bias:
            self.sink_bias = nn.Parameter(torch.zeros(n_heads))
        else:
            self.register_parameter("sink_bias", None)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.d_head,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )

        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, _ = input.shape

        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        # Reshape to (B, n_heads/n_kv_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rotary_emb(T, device=input.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand Key and Value heads for GQA if n_kv_heads < n_heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.attn_scale  # (B, n_heads, T, T)

        # Build causal mask (sliding window or full)
        row_idx = torch.arange(T, device=input.device).unsqueeze(1)
        col_idx = torch.arange(T, device=input.device).unsqueeze(0)
        if self.sliding_window is not None:
            causal_mask = (row_idx >= col_idx) & ((row_idx - col_idx) < self.sliding_window)
        else:
            causal_mask = (row_idx >= col_idx)

        scores = scores.masked_fill(~causal_mask, float("-inf"))

        if self.sink_bias is not None:
            # Learned attention sink in denominator:
            # att_ij = exp(S_ij) / (sum_k exp(S_ik) + exp(sink_h))
            sink = self.sink_bias.view(1, self.n_heads, 1, 1)
            max_score = torch.max(scores, dim=-1, keepdim=True).values
            max_val = torch.maximum(max_score, sink)

            exp_scores = torch.exp(scores - max_val)
            exp_scores = exp_scores.masked_fill(~causal_mask, 0.0)
            exp_sink = torch.exp(sink - max_val)

            denom = exp_scores.sum(dim=-1, keepdim=True) + exp_sink + 1e-12
            att = exp_scores / denom
        else:
            att = torch.softmax(scores, dim=-1)

        att = self.attn_dropout(att)

        out = att @ v  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)

        return self.resid_dropout(self.out_proj(out))
