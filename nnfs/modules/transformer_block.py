import torch
import torch.nn as nn

from nnfs.activations import GELU, ReLU, SwiGLU
from nnfs.layers import CausalMultiHeadAttention, GroupedQueryAttention, LayerNorm, MLP, RMSNorm, SwiGLUMLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_kv_heads: int | None = None,
        attn_type: str = "mha",
        dropout: float = 0.1,
        norm_first: bool = False,
        activation: str | nn.Module = "relu",
        use_rope: bool = False,
        max_position_embeddings: int = 2048,
        norm_type: str = "layernorm",
        bias: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first
        attn_type_str = attn_type.lower() if isinstance(attn_type, str) else "mha"
        if n_kv_heads is None:
            if attn_type_str == "mqa":
                n_kv_heads = 1
            elif attn_type_str == "gqa":
                n_kv_heads = max(1, n_heads // 2)
            else:
                n_kv_heads = n_heads

        if attn_type_str == "mha" and n_kv_heads == n_heads:
            self.attn = CausalMultiHeadAttention(
                d_model,
                n_heads,
                dropout=dropout,
                use_rope=use_rope,
                max_position_embeddings=max_position_embeddings,
                bias=bias,
            )
        else:
            self.attn = GroupedQueryAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                use_rope=use_rope,
                max_position_embeddings=max_position_embeddings,
                bias=bias,
            )

        if isinstance(activation, str):
            act_str = activation.lower()
            if act_str == "relu":
                self.ffn = MLP(d_model, d_ff, ReLU(), dropout)
            elif act_str == "gelu":
                self.ffn = MLP(d_model, d_ff, GELU(), dropout)
            elif act_str == "swiglu":
                self.ffn = SwiGLUMLP(d_model, d_ff, dropout=dropout, bias=False)
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
        elif isinstance(activation, (SwiGLU, SwiGLUMLP)):
            self.ffn = SwiGLUMLP(d_model, d_ff, dropout=dropout, bias=False)
        elif isinstance(activation, nn.Module):
            self.ffn = MLP(d_model, d_ff, activation, dropout)
        else:
            raise ValueError(f"Invalid activation type: {type(activation)}")

        norm_str = norm_type.lower()
        if norm_str == "layernorm":
            self.ln1 = LayerNorm(d_model)
            self.ln2 = LayerNorm(d_model)
        elif norm_str == "rmsnorm":
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, input: torch.Tensor, alibi_bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            x = input + self.attn(self.ln1(input), alibi_bias=alibi_bias)
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(input + self.attn(input, alibi_bias=alibi_bias))
            x = self.ln2(x + self.ffn(x))
        return x
