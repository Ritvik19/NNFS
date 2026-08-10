import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) as introduced by Zhang & Sennrich (2019).

    Computes: Output = (input / sqrt(mean(input^2, dim=-1, keepdim=True) + eps)) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        variance = input.pow(2).mean(dim=-1, keepdim=True)
        return input * torch.rsqrt(variance + self.eps) * self.gamma
