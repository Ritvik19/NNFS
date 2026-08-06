# ReLU (Rectified Linear Unit)

Documentation for the `ReLU` activation function implemented in `NNFS`.

---

## 💡 Overview

The **Rectified Linear Unit (ReLU)** is a classic piecewise linear activation function defined as the positive part of its argument. In `NNFS`, `ReLU` is implemented ground-up using `torch.maximum` on zero tensors.

Module Location: [`nnfs/activations/relu.py`](../../nnfs/activations/relu.py)

---

## 📐 Mathematical Formulation

Given input tensor $x$:

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$$

### Derivative

$$\frac{d}{dx} \text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$

---

## 💻 Ground-Up Implementation

```python
import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.zeros_like(input), input)
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

- **Sparsity**: Produces true zeros for all negative inputs, creating sparse representations.
- **Computational Efficiency**: Evaluates extremely fast with simple element-wise thresholding.
- **Dying ReLU Problem**: Neurons can permanently output zero if large negative gradients shift their bias such that they never activate again.
