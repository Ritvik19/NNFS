import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import Linear
from .swiglu_mlp import SwiGLUMLP


class TopKRouter(nn.Module):
    """Top-K Router Network for Sparse Mixture-of-Experts.

    Given input tensor x of shape (B, T, d_model), computes gating logits
    via a linear projection W_g (shape: d_model -> num_experts), selects the top-K
    experts per token, and returns the top-K routing weights (softmax normalized
    over the top-K experts) and top-K expert indices.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = Linear(d_model, num_experts, bias=bias)
        self.last_router_logits: torch.Tensor | None = None
        self.last_top_k_indices: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)  # (B, T, num_experts)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)  # (B, T, K)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B, T, K)
        self.last_router_logits = logits
        self.last_top_k_indices = top_k_indices
        return top_k_weights, top_k_indices


class SparseMoE(nn.Module):
    """Sparse Mixture-of-Experts (MoE) Layer.

    Replaces a standard FFN layer with N experts (SwiGLUMLP) and a Top-K router.
    For each token, routes the state to the selected top-K experts and accumulates
    their weighted outputs:
        y = sum_{k=1}^K routing_weight_k * Expert_k(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k_experts: int = 2,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.router = TopKRouter(
            d_model, num_experts=num_experts, top_k=top_k_experts, bias=bias
        )
        self.experts = nn.ModuleList(
            [
                SwiGLUMLP(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        top_k_weights, top_k_indices = self.router(x)  # (B, T, K), (B, T, K)

        x_flat = x.view(-1, D)
        weights_flat = top_k_weights.view(-1, self.top_k_experts)  # (B*T, K)
        indices_flat = top_k_indices.view(-1, self.top_k_experts)  # (B*T, K)

        out_flat = torch.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = indices_flat == expert_idx
            if not expert_mask.any():
                continue

            token_indices, k_indices = torch.where(expert_mask)
            expert_inputs = x_flat[token_indices]
            expert_outputs = expert(expert_inputs)

            gating_weights = weights_flat[token_indices, k_indices].unsqueeze(-1)
            weighted_outputs = expert_outputs * gating_weights

            out_flat.index_add_(0, token_indices, weighted_outputs)

        return out_flat.view(B, T, D)
