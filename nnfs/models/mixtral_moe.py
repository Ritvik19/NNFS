import os
import torch
import torch.nn as nn

from nnfs.layers import Dropout, Embedding, Linear, RMSNorm, TiedLinear
from nnfs.modules import MixtralTransformerBlock


class MixtralMoEConfig:
    def __init__(
        self,
        vocab_size: int = 256,
        block_size: int = 1024,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        n_kv_heads: int | None = 2,
        num_experts: int = 8,
        top_k_experts: int = 2,
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
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
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


class MixtralMoE(nn.Module):
    """Mixtral MoE decoder-only language model.

    Based on Jiang et al. (2024): "Mixtral of Experts".
    Key Features:
    - Pre-normalization using RMSNorm
    - Grouped-Query Attention (GQA) across all model sizes
    - Sparse Mixture-of-Experts (MoE) sub-layer (8 SwiGLU experts per layer, Top-2 routing)
    - Rotary Position Embeddings (RoPE) with base theta = 1,000,000.0
    - Weight-tied linear classification head
    """

    def __init__(self, config: MixtralMoEConfig | None = None):
        super().__init__()
        if config is None:
            config = MixtralMoEConfig()
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
                MixtralTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    d_ff=config.d_ff,
                    num_experts=config.num_experts,
                    top_k_experts=config.top_k_experts,
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

    def count_active_parameters(self) -> int:
        """Returns active parameter count executed per token forward pass."""
        active = sum(p.numel() for p in self.tok_embed.parameters())
        active += sum(p.numel() for p in self.rms_f.parameters())

        for block in self.blocks:
            active += sum(p.numel() for p in block.rms_1.parameters())
            active += sum(p.numel() for p in block.attn.parameters())
            active += sum(p.numel() for p in block.rms_2.parameters())
            active += sum(p.numel() for p in block.moe.router.parameters())
            single_expert_params = sum(
                p.numel() for p in block.moe.experts[0].parameters()
            )
            active += self.config.top_k_experts * single_expert_params

        return active

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, Linear):
            if hasattr(module, "weights") and isinstance(module.weights, nn.Parameter):
                nn.init.normal_(module.weights, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            nn.init.normal_(module.embed, mean=0.0, std=0.02)

    def get_router_outputs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Collects cached (router_logits, top_k_indices) from all SparseMoE blocks in the model."""
        outputs = []
        for block in self.blocks:
            if hasattr(block, "moe") and hasattr(block.moe, "router"):
                router = block.moe.router
                if (
                    hasattr(router, "last_router_logits")
                    and router.last_router_logits is not None
                    and hasattr(router, "last_top_k_indices")
                    and router.last_top_k_indices is not None
                ):
                    outputs.append((router.last_router_logits, router.last_top_k_indices))
        return outputs

    def forward(
        self,
        idx: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        t = idx.size(1)

        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
            )

        tok_emb = self.tok_embed(idx)
        x = self.drop(tok_emb)
        for block in self.blocks:
            x = block(x)
        x = self.rms_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss
        return logits

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.config, os.path.join(save_directory, "config.pth"))
        torch.save(self.state_dict(), os.path.join(save_directory, "model.pth"))

    def load_pretrained(self, load_directory: str, map_location=None) -> None:
        state_dict = torch.load(
            os.path.join(load_directory, "model.pth"),
            map_location=map_location,
            weights_only=True,
        )
        self.load_state_dict(state_dict)
