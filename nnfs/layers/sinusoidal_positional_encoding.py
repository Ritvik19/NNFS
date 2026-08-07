import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x_or_positions: torch.Tensor | int) -> torch.Tensor:
        if isinstance(x_or_positions, int):
            return self.pe[:x_or_positions]
        elif x_or_positions.dim() == 1:
            return self.pe[x_or_positions]
        else:
            # Assumes input tensor of shape (B, T) or (B, T, C)
            seq_len = x_or_positions.size(1)
            return self.pe[:seq_len]
