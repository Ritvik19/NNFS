import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import Linear
from .sparse_moe import TopKRouter
from .swiglu_mlp import SwiGLUMLP


class SharedSparseMoE(nn.Module):
    """Llama 4 Shared-and-Routed Sparse Mixture-of-Experts (MoE) Layer.

    In Llama 4 (Scout and Maverick), every token is processed by:
    1. A universal shared expert (SwiGLU MLP) that acts on all tokens unconditionally.
    2. A Top-K router (default K=1) that routes tokens to specialized routed experts
       (SwiGLU MLPs) weighted by their softmax gating probabilities.

    The total layer output is:
        y = SharedExpert(x) + sum_{k=1}^K routing_weight_k * RoutedExpert_k(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_ff_shared: int | None = None,
        num_experts: int = 16,
        top_k_experts: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
        clamp_limit: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_ff_shared = d_ff_shared if d_ff_shared is not None else d_ff
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.clamp_limit = clamp_limit

        # Universal shared expert evaluated on every token
        self.shared_expert = SwiGLUMLP(
            d_model=d_model,
            d_ff=self.d_ff_shared,
            dropout=dropout,
            bias=bias,
            clamp_limit=clamp_limit,
        )

        # Router for selecting top-K routed experts
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k_experts,
            bias=bias,
        )

        # Bank of specialized routed experts
        self.routed_experts = nn.ModuleList(
            [
                SwiGLUMLP(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    bias=bias,
                    clamp_limit=clamp_limit,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # 1. Unconditional Shared Expert computation
        shared_out = self.shared_expert(x)  # (B, T, D)

        # 2. Top-K Routed Experts computation
        top_k_weights, top_k_indices = self.router(x)  # (B, T, K), (B, T, K)

        x_flat = x.view(-1, D)
        weights_flat = top_k_weights.view(-1, self.top_k_experts)  # (B*T, K)
        indices_flat = top_k_indices.view(-1, self.top_k_experts)  # (B*T, K)

        routed_out_flat = torch.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.routed_experts):
            expert_mask = indices_flat == expert_idx
            if not expert_mask.any():
                continue

            token_indices, k_indices = torch.where(expert_mask)
            expert_inputs = x_flat[token_indices]
            expert_outputs = expert(expert_inputs)

            gating_weights = weights_flat[token_indices, k_indices].unsqueeze(-1)
            weighted_outputs = expert_outputs * gating_weights

            routed_out_flat.index_add_(0, token_indices, weighted_outputs)

        routed_out = routed_out_flat.view(B, T, D)

        # 3. Combine shared and routed outputs
        return shared_out + routed_out
