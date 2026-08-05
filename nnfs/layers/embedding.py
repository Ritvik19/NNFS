import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(Embedding, self).__init__()
        self.embed = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embed[input]