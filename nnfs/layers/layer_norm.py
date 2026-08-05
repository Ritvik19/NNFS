import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(dim=-1, keepdim=True)
        var = ((input - mean) ** 2).mean(dim=-1, keepdim=True)
        return (input - mean) / torch.sqrt(var + 1e-5) * self.gamma + self.beta