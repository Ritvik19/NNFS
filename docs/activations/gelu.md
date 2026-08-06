# GELU (Gaussian Error Linear Unit)

Documentation for the `GELU` activation function implemented in `NNFS`.

---

## 💡 Overview

The **Gaussian Error Linear Unit (GELU)** was introduced by [Hendrycks & Gimpel (2016)](https://arxiv.org/abs/1606.08415) and is the default activation function in landmark Transformer models such as **GPT-1**, **GPT-2**, and **BERT**. 

Unlike ReLU, GELU weights inputs by their value rather than gating inputs strictly by sign, providing a smooth, non-monotonic curve with probabilistic properties.

Module Location: [`nnfs/activations/gelu.py`](../../nnfs/activations/gelu.py)

---

## 📐 Mathematical Formulation

### Exact Definition

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x)$$

where $X \sim \mathcal{N}(0, 1)$ is the standard normal cumulative distribution function.

### Tanh Approximation

In `NNFS` (following OpenAI GPT-1 and GPT-2 specifications), the fast tanh approximation is implemented from scratch:

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right)\right)$$

---

## 💻 Ground-Up Implementation

```python
import math
import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
```

---

## 📊 Tensor Shapes & Parameters

| Attribute | Specification |
|---|---|
| **Input Shape** | $(B, \dots, D)$ where $B$ is batch size and $D$ is feature dimension |
| **Output Shape** | $(B, \dots, D)$ (Identical to input shape) |
| **Learnable Parameters** | 0 |
| **State Buffers** | 0 |

---

## ⚙️ Key Characteristics

- **Smooth Differentiability**: Continuous derivative across all real values with no abrupt corners (unlike ReLU).
- **Non-Monotonicity**: Has a subtle dip in the range $x \in (-0.1, 0.0)$, allowing small negative activations to propagate gradient signal.
- **Probabilistic Intuition**: Acts as a stochastic drop-path where inputs are zeroed with probability $1 - \Phi(x)$.
