import os
import torch
import torch.nn as nn

from nnfs.layers import Dropout, Embedding, Linear, RMSNorm, TiedLinear
from nnfs.modules import MistralTransformerBlock


class MistralConfig:
    def __init__(
        self,
        vocab_size: int = 256,
        block_size: int = 1024,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        n_kv_heads: int | None = 2,
        d_ff: int | None = 1024,
        dropout: float = 0.1,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
        sliding_window: int | None = 4096,
        interleaved_sliding_window: bool = False,
        eps: float = 1e-5,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        if d_ff is None:
            raw_d_ff = int(2 * 4 * d_model / 3)
            self.d_ff = ((raw_d_ff + 255) // 256) * 256
        else:
            self.d_ff = d_ff
        self.dropout = float(dropout)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.sliding_window = int(sliding_window) if sliding_window is not None else None
        self.interleaved_sliding_window = bool(interleaved_sliding_window)
        self.eps = float(eps)


class Mistral(nn.Module):
    """Mistral decoder-only language model.

    Based on Jiang et al. (2023): "Mistral 7B".
    Key Features:
    - Pre-normalization using RMSNorm
    - Grouped-Query Attention (GQA) across all model sizes
    - Sliding Window Attention (SWA) with optional interleaved layer configuration
    - SwiGLU activation with d_ff = multiple_of_256(2/3 * 4 * d_model)
    - Rotary Position Embeddings (RoPE) with configurable base theta (default = 1,000,000.0)
    - Tied linear classification head (miniature implementation mode)
    - Bias-free linear projections
    """

    def __init__(self, config: MistralConfig | None = None):
        super().__init__()
        if config is None:
            config = MistralConfig()
        self.config = config
        self.tok_embed = Embedding(config.vocab_size, config.d_model)
        self.drop = Dropout(config.dropout)

        blocks = []
        for i in range(config.n_layers):
            if config.interleaved_sliding_window:
                sw = config.sliding_window if (i % 2 == 0) else None
            else:
                sw = config.sliding_window

            blocks.append(
                MistralTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    max_position_embeddings=config.block_size,
                    rope_theta=config.rope_theta,
                    rope_scaling=config.rope_scaling,
                    sliding_window=sw,
                    eps=config.eps,
                )
            )
        self.blocks = nn.ModuleList(blocks)
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
