import torch
import torch.nn as nn

from nnfs.activations import SwiGLU
from .dropout import Dropout
from .linear import Linear


class SwiGLUMLP(nn.Module):
    """Feed-Forward Network with SwiGLU activation as used in PaLM.

    Computes: Output = Dropout( ( Swish(x * W_gate) * (x * W_up) ) * W_down )
    where linear transformations have no bias by default.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = False,
        clamp_limit: float | None = None,
    ):
        super().__init__()
        self.w_gate = Linear(d_model, d_ff, bias=bias)
        self.w_up = Linear(d_model, d_ff, bias=bias)
        self.w_down = Linear(d_ff, d_model, bias=bias)
        self.act = SwiGLU(clamp_limit=clamp_limit)
        self.dropout = Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(input)
        up = self.w_up(input)
        hidden = self.act(gate, up)
        out = self.w_down(hidden)
        return self.dropout(out)
