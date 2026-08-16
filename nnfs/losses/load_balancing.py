import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple


class LoadBalancingLoss(nn.Module):
    """Auxiliary Load Balancing Loss for Sparse Mixture-of-Experts (MoE).

    Computes the auxiliary load balancing loss across experts as introduced in Switch
    Transformers (Fedus et al., 2021) and Mixtral of Experts (Jiang et al., 2024):

        L_aux = num_experts * sum_{i=1}^{num_experts} (f_i * P_i)

    where:
        f_i: fraction of tokens assigned to expert i (mean over top-k selections)
        P_i: mean gating probability assigned to expert i (mean of softmax probabilities)
    """

    def __init__(self, num_experts: Union[int, None] = None):
        super().__init__()
        self.num_experts = num_experts

    def _compute_layer_loss(
        self, router_logits: torch.Tensor, top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """Computes load balancing loss for a single MoE layer.

        Args:
            router_logits: Tensor of shape (..., num_experts) containing unnormalized router gating logits.
            top_k_indices: Tensor of shape (..., top_k) containing indices of top-K selected experts.

        Returns:
            Scalar tensor containing the layer auxiliary loss.
        """
        num_experts = router_logits.size(-1)
        top_k = top_k_indices.size(-1)

        # Flatten batch and sequence dimensions: (N_tokens, num_experts) and (N_tokens, top_k)
        logits_flat = router_logits.reshape(-1, num_experts)
        indices_flat = top_k_indices.reshape(-1, top_k)

        # Compute P_i: mean softmax gating probability assigned to expert i across all tokens
        router_probs = F.softmax(logits_flat, dim=-1)  # (N_tokens, num_experts)
        p_i = router_probs.mean(dim=0)  # (num_experts,)

        # Compute f_i: fraction of token assignments routed to expert i
        # One-hot mask: (N_tokens, top_k, num_experts)
        one_hot = F.one_hot(indices_flat, num_classes=num_experts).float()
        # Sum over top_k dimension to get count per token: (N_tokens, num_experts)
        assignments = one_hot.sum(dim=1)
        f_i = assignments.mean(dim=0) / float(top_k)  # (num_experts,)

        # Aux loss = num_experts * sum(f_i * P_i)
        loss = num_experts * torch.sum(f_i * p_i)
        return loss

    def forward(
        self,
        router_outputs: Union[
            Tuple[torch.Tensor, torch.Tensor],
            List[Tuple[torch.Tensor, torch.Tensor]],
        ],
    ) -> torch.Tensor:
        """
        Args:
            router_outputs: Single tuple (router_logits, top_k_indices) or list of layer tuples
                           [(router_logits_1, top_k_indices_1), ...].

        Returns:
            Scalar tensor representing the total or average load balancing loss across layers.
        """
        if isinstance(router_outputs, tuple):
            return self._compute_layer_loss(router_outputs[0], router_outputs[1])

        if isinstance(router_outputs, list):
            if not router_outputs:
                raise ValueError("router_outputs list cannot be empty.")
            layer_losses = [
                self._compute_layer_loss(logits, indices)
                for logits, indices in router_outputs
            ]
            return torch.stack(layer_losses).mean()

        raise TypeError(
            f"Expected tuple or list of tuples for router_outputs, got {type(router_outputs)}"
        )
