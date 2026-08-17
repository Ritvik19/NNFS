import math
import torch
import torch.nn as nn

from .dropout import Dropout
from .linear import Linear
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class Llama4Attention(nn.Module):
    """Llama 4 Grouped-Query Attention layer with iRoPE support and temperature scaling.

    Key Features:
    - Grouped-Query Attention (GQA) across all model sizes
    - iRoPE Support:
      * When is_rope_layer=True: Applies Rotary Positional Embeddings (RoPE) with
        chunked local causal attention (chunk_size).
      * When is_rope_layer=False (NoPE layer): Disables positional embeddings and uses
        global full-context causal attention.
    - Inference-Time Attention Temperature Scaling: Modulates query-key scaling
      for length generalization over massive context windows.
    - Bias-free linear projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        d_head: int | None = None,
        dropout: float = 0.0,
        is_rope_layer: bool = True,
        chunk_size: int | None = 8192,
        max_position_embeddings: int = 8192,
        rope_theta: float = 500000.0,
        rope_scaling: dict | None = None,
        temp_scaling: float = 1.0,
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
        self.d_head = d_head if d_head is not None else (d_model // n_heads)
        self.is_rope_layer = is_rope_layer
        self.chunk_size = chunk_size
        self.temp_scaling = float(temp_scaling) if temp_scaling is not None else 1.0

        # Scale dot products by 1 / (sqrt(d_head) * temp_scaling)
        self.attn_scale = 1.0 / (math.sqrt(self.d_head) * self.temp_scaling)

        self.q_proj = Linear(d_model, n_heads * self.d_head, bias=bias)
        self.k_proj = Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.v_proj = Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.out_proj = Linear(n_heads * self.d_head, d_model, bias=bias)

        if self.is_rope_layer:
            self.rotary_emb = RotaryEmbedding(
                self.d_head,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )
        else:
            self.rotary_emb = None

        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)

    def forward(self, input: torch.Tensor, alibi_bias: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = input.shape

        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        # Reshape to (B, n_heads/n_kv_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply RoPE only in RoPE-designated layers (iRoPE mechanism)
        if self.is_rope_layer and self.rotary_emb is not None:
            cos, sin = self.rotary_emb(T, device=input.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand Key and Value heads for GQA if n_kv_heads < n_heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.attn_scale  # (B, n_heads, T, T)
        if alibi_bias is not None:
            scores = scores + alibi_bias

        row_idx = torch.arange(T, device=input.device).unsqueeze(1)
        col_idx = torch.arange(T, device=input.device).unsqueeze(0)

        # Chunked local causal attention for RoPE layers, global full causal attention for NoPE layers
        if self.is_rope_layer and self.chunk_size is not None:
            causal_mask = (row_idx >= col_idx) & ((row_idx - col_idx) < self.chunk_size)
        else:
            causal_mask = (row_idx >= col_idx)

        scores = scores.masked_fill(~causal_mask, float("-inf"))

        att = torch.softmax(scores, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)

        return self.resid_dropout(self.out_proj(out))
