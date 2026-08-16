import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """SwiGLU activation function built ground-up from mathematical primitives.

    Computes:
        SwiGLU(gate, up) = Swish(clamp(gate, -limit, limit)) * up
    where Swish(z) = z * sigmoid(z) = z / (1 + exp(-z)).
    """

    def __init__(self, beta: float = 1.0, clamp_limit: float | None = None):
        super().__init__()
        self.beta = beta
        self.clamp_limit = clamp_limit

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.clamp_limit is not None:
            gate = torch.clamp(gate, min=-self.clamp_limit, max=self.clamp_limit)
        # Ground-up sigmoid: 1 / (1 + exp(-beta * gate))
        sigmoid_gate = 1.0 / (1.0 + torch.exp(-self.beta * gate))
        swish_gate = gate * sigmoid_gate
        return swish_gate * up

