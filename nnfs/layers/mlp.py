import torch
import torch.nn as nn

from .dropout import Dropout
from .linear import Linear


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = Dropout(dropout)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(input))))
