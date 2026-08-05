import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return input
        keep = torch.rand_like(input) >= self.p
        return input * keep / (1.0 - self.p)
