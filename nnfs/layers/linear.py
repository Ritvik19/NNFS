import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_size))
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.matmul(input, self.weights) + (self.bias if self.bias is not None else 0)  