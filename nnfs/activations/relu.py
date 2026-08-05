import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.zeros_like(input), input)