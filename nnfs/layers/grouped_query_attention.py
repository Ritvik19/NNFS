import math
import torch
import torch.nn as nn

from .dropout import Dropout
from .linear import Linear
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE).

    Used in LLaMA 2 (34B, 70B) where Query heads (n_heads) are partitioned
    into groups sharing a smaller number of Key/Value heads (n_kv_heads).

    Supports:
    - Multi-Head Attention (MHA) when n_kv_heads == n_heads
    - Multi-Query Attention (MQA) when n_kv_heads == 1
    - Grouped-Query Attention (GQA) when 1 < n_kv_heads < n_heads
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        bias: bool = False,
    ):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.attn_scale = 1.0 / math.sqrt(self.d_head)
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.v_proj = Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.out_proj = Linear(d_model, d_model, bias=bias)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.d_head,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )

        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)

    def forward(self, input: torch.Tensor, alibi_bias: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = input.shape

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

        # Expand Key and Value heads for GQA/MQA if n_kv_heads < n_heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.attn_scale  # (B, n_heads, T, T)
        if alibi_bias is not None:
            scores = scores + alibi_bias

        causal_mask = torch.tril(torch.ones(T, T, device=input.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        att = torch.softmax(scores, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))
