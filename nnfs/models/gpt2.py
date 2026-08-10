import torch
import torch.nn as nn

from nnfs.layers import Dropout, Embedding, LayerNorm, Linear, TiedLinear
from nnfs.modules import GPT2TransformerBlock


class GPT2Config:
    def __init__(
        self,
        vocab_size: int = 256,
        block_size: int = 1024,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config | None = None):
        super().__init__()
        if config is None:
            config = GPT2Config()
        self.config = config
        self.tok_embed = Embedding(config.vocab_size, config.d_model)
        self.pos_embed = Embedding(config.block_size, config.d_model)
        self.drop = Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                GPT2TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = LayerNorm(config.d_model)
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

        if position_ids is None:
            pos = torch.arange(T, device=idx.device)
        else:
            pos = position_ids

        x = self.drop(self.tok_embed(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def save_pretrained(self, save_path: str) -> None:
        torch.save(self.config, save_path + "/config.pth")
        torch.save(self.state_dict(), save_path + "/model.pth")

    def load_pretrained(self, load_path: str, map_location: str | torch.device | None = None) -> None:
        self.config = torch.load(
            load_path + "/config.pth",
            map_location=map_location,
            weights_only=False,
        )
        self.load_state_dict(
            torch.load(
                load_path + "/model.pth",
                map_location=map_location,
                weights_only=True,
            )
        )
