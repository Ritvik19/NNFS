# LayerNorm (Layer Normalization)

Documentation for the `LayerNorm` layer implemented in `NNFS`.

---

## 💡 Overview

**Layer Normalization** ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)) standardizes the activations across feature dimensions for each individual sample in a batch. In `NNFS`, `LayerNorm` is implemented ground-up from mathematical mean, variance, and square root operations.

Module Location: [`nnfs/layers/layer_norm.py`](../../nnfs/layers/layer_norm.py)

---

## 📐 Mathematical Formulation

Given input vector $x \in \mathbb{R}^{d_{\text{model}}}$, learnable gain parameter $\gamma \in \mathbb{R}^{d_{\text{model}}}$, and learnable bias parameter $\beta \in \mathbb{R}^{d_{\text{model}}}$:

1. **Feature Mean**:
   $$\mu = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} x_i$$

2. **Feature Variance**:
   $$\sigma^2 = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (x_i - \mu)^2$$

3. **Normalization & Rescaling**:
   $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
   $$y_i = \hat{x}_i \cdot \gamma_i + \beta_i$$

where $\epsilon = 10^{-5}$ prevents division by zero.

---

## 💻 Ground-Up Implementation

```python
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
```

---

## 📊 Tensor Shapes & Parameters

| Parameter / Tensor | Symbol / Shape | Description |
|---|---|---|
| **Input Shape** | $(B, T, d_{\text{model}})$ | Input hidden state |
| **Output Shape** | $(B, T, d_{\text{model}})$ | Normalized output state |
| **Gamma ($\gamma$)** | $(d_{\text{model}})$ | Learnable gain scaling vector (initialized to $1.0$) |
| **Beta ($\beta$)** | $(d_{\text{model}})$ | Learnable shift bias vector (initialized to $0.0$) |

### Parameter Formula

$$\text{Params} = 2 \cdot d_{\text{model}}$$
