import os
import torch
import torch.nn as nn

from nnfs.layers import Dropout, Embedding, Linear, RMSNorm, TiedLinear
from nnfs.modules import Llama1TransformerBlock


class Llama1Config:
    def __init__(
        self,
        vocab_size: int = 256,
        block_size: int = 1024,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int | None = 1024,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        if d_ff is None:
            raw_d_ff = int(2 * 4 * d_model / 3)
            self.d_ff = ((raw_d_ff + 255) // 256) * 256
        else:
            self.d_ff = d_ff
        self.dropout = dropout
        self.eps = eps


class Llama1(nn.Module):
    """LLaMA 1 miniature decoder-only language model.

    Based on Touvron et al. (2023): "LLaMA: Open and Efficient Foundation Language Models".
    Key Features:
    - Pre-normalization using RMSNorm
    - SwiGLU activation with d_ff = multiple_of_256(2/3 * 4 * d_model)
    - Rotary Position Embeddings (RoPE)
    - Bias-free linear layers
    """

    def __init__(self, config: Llama1Config | None = None):
        super().__init__()
        if config is None:
            config = Llama1Config()
        self.config = config
        self.tok_embed = Embedding(config.vocab_size, config.d_model)
        self.drop = Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                Llama1TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    max_position_embeddings=config.block_size,
                    eps=config.eps,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.rms_f = RMSNorm(config.d_model, eps=config.eps)
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

        x = self.drop(self.tok_embed(idx))
        for block in self.blocks:
            x = block(x)
        x = self.rms_f(x)
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
