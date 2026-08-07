import math
import os

import torch
import torch.nn as nn

from nnfs.layers import Dropout, Embedding, LayerNorm, Linear, SinusoidalPositionalEncoding, TiedLinear
from nnfs.modules import VaswaniTransformerBlock


class VaswaniDecoderOnlyConfig:
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
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.norm_first = norm_first


class VaswaniDecoderOnly(nn.Module):
    def __init__(self, config: VaswaniDecoderOnlyConfig):
        super().__init__()
        self.config = config
        self.tok_embed = Embedding(config.vocab_size, config.d_model)
        self.pos_embed = SinusoidalPositionalEncoding(config.block_size, config.d_model)
        self.drop = Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                VaswaniTransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    norm_first=config.norm_first,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = LayerNorm(config.d_model) if config.norm_first else None
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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"

        # Vaswani et al. Section 3.4: Multiply token embeddings by sqrt(d_model) before adding positional encoding
        tok_emb = self.tok_embed(idx) * math.sqrt(self.config.d_model)
        pos_emb = self.pos_embed(T)

        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)

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
