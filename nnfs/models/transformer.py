import math
import os

import torch
import torch.nn as nn

from nnfs.layers import (
    ALiBiPositionalBias,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    SinusoidalPositionalEncoding,
    TiedLinear,
)
from nnfs.modules import TransformerBlock


class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        block_size: int = 512,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False,
        positional_encoding: str = "sinusoidal",
        activation: str = "relu",
        norm_type: str = "layernorm",
        bias: bool = True,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.norm_first = norm_first
        self.positional_encoding = positional_encoding.lower()
        self.activation = activation.lower()
        self.norm_type = norm_type.lower()
        self.bias = bias


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_embed = Embedding(config.vocab_size, config.d_model)

        pos_enc = config.positional_encoding
        if pos_enc == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEncoding(config.block_size, config.d_model)
        elif pos_enc in ("learned", "absolute"):
            self.pos_embed = Embedding(config.block_size, config.d_model)
        elif pos_enc == "alibi":
            self.pos_embed = ALiBiPositionalBias(config.n_heads, max_seq_len=config.block_size)
        elif pos_enc in ("rope", "none"):
            self.pos_embed = None
        else:
            raise ValueError(f"Unsupported positional_encoding: {config.positional_encoding}")

        use_rope = pos_enc == "rope"
        self.drop = Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    norm_first=config.norm_first,
                    activation=config.activation,
                    use_rope=use_rope,
                    max_position_embeddings=config.block_size,
                    norm_type=config.norm_type,
                    bias=config.bias,
                )
                for _ in range(config.n_layers)
            ]
        )
        if config.norm_first:
            if config.norm_type == "layernorm":
                self.ln_f = LayerNorm(config.d_model)
            elif config.norm_type == "rmsnorm":
                self.ln_f = RMSNorm(config.d_model)
            else:
                raise ValueError(f"Unsupported norm_type: {config.norm_type}")
        else:
            self.ln_f = None
        self.lm_head = TiedLinear(self.tok_embed, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, Linear):
            if hasattr(module, "weights") and isinstance(module.weights, nn.Parameter):
                nn.init.normal_(module.weights, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            nn.init.normal_(module.embed, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        _, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"

        pos_enc = self.config.positional_encoding
        alibi_bias = None

        if pos_enc == "sinusoidal":
            tok_emb = self.tok_embed(idx) * math.sqrt(self.config.d_model)
            pos_emb = self.pos_embed(T)
            x = self.drop(tok_emb + pos_emb)
        elif pos_enc in ("learned", "absolute"):
            tok_emb = self.tok_embed(idx)
            if position_ids is None:
                pos = torch.arange(T, device=idx.device)
            else:
                pos = position_ids
            pos_emb = self.pos_embed(pos)
            x = self.drop(tok_emb + pos_emb)
        elif pos_enc == "alibi":
            x = self.drop(self.tok_embed(idx))
            alibi_bias = self.pos_embed(T, device=idx.device)
        elif pos_enc in ("rope", "none"):
            x = self.drop(self.tok_embed(idx))
        else:
            raise ValueError(f"Unsupported positional_encoding: {pos_enc}")

        for block in self.blocks:
            x = block(x, alibi_bias=alibi_bias)

        if self.ln_f is not None:
            x = self.ln_f(x)

        return self.lm_head(x)

    def save_pretrained(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.config, os.path.join(save_path, "config.pth"))
        torch.save(self.state_dict(), os.path.join(save_path, "model.pth"))

    def load_pretrained(self, load_path: str, map_location: str | torch.device | None = None) -> None:
        self.config = torch.load(
            os.path.join(load_path, "config.pth"),
            map_location=map_location,
            weights_only=False,
        )
        self.load_state_dict(
            torch.load(
                os.path.join(load_path, "model.pth"),
                map_location=map_location,
                weights_only=True,
            )
        )
