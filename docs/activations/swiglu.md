# SwiGLU (Swish-Gated Linear Unit)

Documentation for the `SwiGLU` activation function implemented in `NNFS`.

---

## 💡 Overview

**SwiGLU (Swish-Gated Linear Unit)** was proposed by [Noam Shazeer (2020)](https://arxiv.org/abs/2002.05202) and is utilized in modern state-of-the-art Transformer architectures including **PaLM**, **LLaMA**, and **Mistral**.

SwiGLU combines the **Swish** activation function ($\text{Swish}_{\beta}(x) = x \cdot \sigma(\beta x)$) with Gated Linear Units (GLU). In `NNFS`, `SwiGLU` is implemented ground-up from mathematical sigmoid primitives without standard library abstractions.

Module Location: [`nnfs/activations/swiglu.py`](../../nnfs/activations/swiglu.py)

---

## 📐 Mathematical Formulation

Given dual input tensors $\text{gate}$ and $\text{up}$:

$$\text{SwiGLU}(\text{gate}, \text{up}) = \text{Swish}_{\beta}(\text{gate}) \odot \text{up}$$

where:

$$\text{Swish}_{\beta}(\text{gate}) = \text{gate} \cdot \sigma(\beta \cdot \text{gate}) = \frac{\text{gate}}{1 + e^{-\beta \cdot \text{gate}}}$$

$$\text{SwiGLU}(\text{gate}, \text{up}) = \left( \frac{\text{gate}}{1 + e^{-\beta \cdot \text{gate}}} \right) \odot \text{up}$$

---

## 💻 Ground-Up Implementation

```python
import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    """SwiGLU activation function built ground-up from mathematical primitives.

    Computes:
        SwiGLU(gate, up) = Swish(gate) * up
    where Swish(z) = z * sigmoid(z) = z / (1 + exp(-z)).
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        # Ground-up sigmoid: 1 / (1 + exp(-beta * gate))
        sigmoid_gate = 1.0 / (1.0 + torch.exp(-self.beta * gate))
        swish_gate = gate * sigmoid_gate
        return swish_gate * up
```

---

## 📊 Tensor Shapes & Parameters

| Attribute | Specification |
|---|---|
| **Gate Tensor Shape** | $(B, \dots, D_{\text{ff}})$ |
| **Up Tensor Shape** | $(B, \dots, D_{\text{ff}})$ |
| **Output Shape** | $(B, \dots, D_{\text{ff}})$ |
| **Learnable Parameters** | 0 |
| **Hyperparameter** | `beta: float = 1.0` |

---

## ⚙️ Key Characteristics

- **Gated Architecture**: One linear projection (`gate`) dynamically filters or gates the information flow of a second linear projection (`up`).
- **Empirical Superiority**: Consistently outperforms standard ReLU and GELU activations in LLM pre-training loss benchmarks.
- **Used in `SwiGLUMLP`**: Integrates seamlessly into parallel feed-forward networks (e.g., in PaLM).
