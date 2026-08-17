import os
import torch
import torch.nn as nn

from nnfs.layers import Dropout, Embedding, Linear, RMSNorm, TiedLinear
from nnfs.modules import Llama4TransformerBlock


class Llama4Config:
    """Configuration class for Llama 4 models (Scout and Maverick)."""

    def __init__(
        self,
        vocab_size: int = 256,
        block_size: int = 1024,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        n_kv_heads: int | None = 2,
        d_head: int | None = None,
        d_ff: int | None = 1024,
        d_ff_shared: int | None = None,
        num_experts: int = 8,
        top_k_experts: int = 1,
        dropout: float = 0.1,
        rope_theta: float = 500000.0,
        rope_scaling: dict | None = None,
        irope_ratio: int = 3,
        chunk_size: int | None = 256,
        temp_scaling: float = 1.0,
        clamp_limit: float | None = None,
        tie_word_embeddings: bool = True,
        eps: float = 1e-5,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.d_head = d_head if d_head is not None else (d_model // n_heads)
        self.d_ff = d_ff if d_ff is not None else (d_model * 4)
        self.d_ff_shared = d_ff_shared if d_ff_shared is not None else self.d_ff
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.dropout = float(dropout)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.irope_ratio = int(irope_ratio)
        self.chunk_size = int(chunk_size) if chunk_size is not None else None
        self.temp_scaling = float(temp_scaling)
        self.clamp_limit = float(clamp_limit) if clamp_limit is not None else None
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.eps = float(eps)

    @classmethod
    def llama_4_scout(cls, **kwargs):
        """Standard Llama 4 Scout full configuration (109B total params, ~17B active)."""
        defaults = dict(
            vocab_size=128256,
            block_size=10485760,  # 10M token context window
            d_model=5120,
            n_layers=48,
            n_heads=40,
            n_kv_heads=8,
            d_head=128,
            d_ff=8192,
            d_ff_shared=8192,
            num_experts=16,
            top_k_experts=1,
            dropout=0.0,
            rope_theta=500000.0,
            irope_ratio=3,
            chunk_size=8192,
            temp_scaling=1.0,
            tie_word_embeddings=False,
            eps=1e-5,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def llama_4_maverick(cls, **kwargs):
        """Standard Llama 4 Maverick full configuration (400B total params, ~17B active)."""
        defaults = dict(
            vocab_size=128256,
            block_size=1048576,  # 1M token context window
            d_model=5120,
            n_layers=48,
            n_heads=40,
            n_kv_heads=8,
            d_head=128,
            d_ff=8192,
            d_ff_shared=8192,
            num_experts=128,
            top_k_experts=1,
            dropout=0.0,
            rope_theta=500000.0,
            irope_ratio=3,
            chunk_size=8192,
            temp_scaling=1.0,
            tie_word_embeddings=False,
            eps=1e-5,
        )
        defaults.update(kwargs)
        return cls(**defaults)


class Llama4(nn.Module):
    """Llama 4 autoregressive Mixture-of-Experts (MoE) language model.

    Based on Meta AI (2025): "The Llama 4 Herd of Models" (Scout & Maverick).

    Key Features:
    - Pre-normalization using RMSNorm
    - Grouped-Query Attention (GQA) with bias-free projections
    - iRoPE (Interleaved Rotary Position Embeddings):
      * 3:1 ratio of Chunked RoPE local attention to Global NoPE full causal attention
    - Inference-Time Attention Temperature Scaling for long-context length generalization
    - Shared-and-Routed Sparse MoE:
      * 1 unconditionally active Shared SwiGLU Expert
      * Top-1 Router selecting from N specialized Routed SwiGLU Experts
    - Configurable tied or untied classification head
    """

    def __init__(self, config: Llama4Config | None = None):
        super().__init__()
        if config is None:
            config = Llama4Config()
        self.config = config

        self.tok_embed = Embedding(config.vocab_size, config.d_model)
        self.drop = Dropout(config.dropout)

        blocks = []
        for i in range(config.n_layers):
            # iRoPE interleaving: if irope_ratio > 0, every (irope_ratio + 1)-th layer is a global NoPE layer
            if config.irope_ratio > 0:
                is_rope = ((i + 1) % (config.irope_ratio + 1)) != 0
            else:
                is_rope = True

            blocks.append(
                Llama4TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    d_head=config.d_head,
                    d_ff=config.d_ff,
                    d_ff_shared=config.d_ff_shared,
                    num_experts=config.num_experts,
                    top_k_experts=config.top_k_experts,
                    dropout=config.dropout,
                    is_rope_layer=is_rope,
                    chunk_size=config.chunk_size,
                    max_position_embeddings=config.block_size,
                    rope_theta=config.rope_theta,
                    rope_scaling=config.rope_scaling,
                    temp_scaling=config.temp_scaling,
                    clamp_limit=config.clamp_limit,
                    eps=config.eps,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.rms_f = RMSNorm(config.d_model, eps=config.eps)

        if config.tie_word_embeddings:
            self.lm_head = TiedLinear(self.tok_embed, bias=False)
        else:
            self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def count_active_parameters(self) -> int:
        """Returns active parameter count executed per token forward pass."""
        active = sum(p.numel() for p in self.tok_embed.parameters())
        active += sum(p.numel() for p in self.rms_f.parameters())
        if not self.config.tie_word_embeddings:
            active += sum(p.numel() for p in self.lm_head.parameters())

        for block in self.blocks:
            active += sum(p.numel() for p in block.rms_1.parameters())
            active += sum(p.numel() for p in block.attn.parameters())
            active += sum(p.numel() for p in block.rms_2.parameters())
            active += sum(p.numel() for p in block.moe.router.parameters())
            active += sum(p.numel() for p in block.moe.shared_expert.parameters())
            single_routed_expert_params = sum(
                p.numel() for p in block.moe.routed_experts[0].parameters()
            )
            active += self.config.top_k_experts * single_routed_expert_params

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
        """Collects cached (router_logits, top_k_indices) from all Llama 4 blocks."""
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
