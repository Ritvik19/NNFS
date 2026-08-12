import math
import torch
import torch.nn as nn

from .dropout import Dropout
from .linear import Linear
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA) with Rotary Position Embeddings (RoPE) as used in PaLM.

    Queries are projected into n_heads distinct heads, while Key and Value
    projections share a single head across all Query heads.
    Linear projections use bias=False by default following PaLM specs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        bias: bool = False,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_scale = 1.0 / math.sqrt(self.d_head)
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, self.d_head, bias=bias)
        self.v_proj = Linear(d_model, self.d_head, bias=bias)
        self.out_proj = Linear(d_model, d_model, bias=bias)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.d_head, max_position_embeddings=max_position_embeddings)
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)

    def forward(self, input: torch.Tensor, alibi_bias: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = input.shape

        q = self.q_proj(input)  # (B, T, d_model)
        k = self.k_proj(input)  # (B, T, d_head)
        v = self.v_proj(input)  # (B, T, d_head)

        # Reshape for multi-head Q and single-head K, V
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.view(B, T, 1, self.d_head).transpose(1, 2)            # (B, 1, T, d_head)
        v = v.view(B, T, 1, self.d_head).transpose(1, 2)            # (B, 1, T, d_head)

        # Apply RoPE to queries and keys if enabled
        if self.use_rope:
            cos, sin = self.rotary_emb(T, device=input.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention with broadcasting over K (1 head -> n_heads)
        att = (q @ k.transpose(-2, -1)) * self.attn_scale  # (B, n_heads, T, T)
        if alibi_bias is not None:
            att = att + alibi_bias

        causal_mask = torch.tril(torch.ones(T, T, device=input.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)

        return self.resid_dropout(self.out_proj(out))
