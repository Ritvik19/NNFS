# RMSNorm (Root Mean Square Normalization)

Documentation for the `RMSNorm` layer implemented in `NNFS`.

---

## 💡 Overview

**Root Mean Square Layer Normalization (RMSNorm)** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) is a computationally efficient variation of Layer Normalization. It assumes that the mean-centering operation in standard LayerNorm can be discarded without harming model performance, scaling inputs strictly by their root mean square. In `NNFS`, `RMSNorm` is implemented from scratch as used in landmark architectures such as LLaMA and PaLM.

Module Location: [`nnfs/layers/rms_norm.py`](../../nnfs/layers/rms_norm.py)

---

## 📐 Mathematical Formulation

Given input vector $x \in \mathbb{R}^{d_{\text{model}}}$ and learnable scale parameter $\gamma \in \mathbb{R}^{d_{\text{model}}}$:

1. **Root Mean Square**:
   $$\text{RMS}(x) = \sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} x_i^2 + \epsilon}$$

2. **Normalization & Rescaling**:
   $$\bar{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$

where $\epsilon = 10^{-5}$ prevents division by zero.

---

## 💻 Ground-Up Implementation

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        variance = input.pow(2).mean(dim=-1, keepdim=True)
        return input * torch.rsqrt(variance + self.eps) * self.gamma
```

---

## 📊 Tensor Shapes & Parameters

| Parameter / Tensor | Symbol / Shape | Description |
|---|---|---|
| **Input Shape** | $(B, T, d_{\text{model}})$ | Input hidden state |
| **Output Shape** | $(B, T, d_{\text{model}})$ | Normalized output state |
| **Gamma ($\gamma$)** | $(d_{\text{model}})$ | Learnable gain scaling vector (initialized to $1.0$) |

### Parameter Formula

$$\text{Params} = 1 \cdot d_{\text{model}}$$
