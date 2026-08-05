import torch
import torch.nn as nn

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(output, dim=-1)
        loss = -torch.log(probs[torch.arange(len(output)), target] + 1e-12).mean()
        return loss
        
