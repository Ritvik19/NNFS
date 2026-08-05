import math

import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))