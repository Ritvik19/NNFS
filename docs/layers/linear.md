# Linear Layer

Documentation for the `Linear` projection layer implemented in `NNFS`.

---

## 💡 Overview

The `Linear` layer performs an affine linear transformation on incoming tensor data using weight matrix multiplication and optional bias addition. In `NNFS`, `Linear` is built ground-up with explicit `nn.Parameter` tensors and `torch.matmul`.

Module Location: [`nnfs/layers/linear.py`](../../nnfs/layers/linear.py)

---

## 📐 Mathematical Formulation

Given input matrix $X \in \mathbb{R}^{B \times T \times d_{\text{in}}}$, weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, and optional bias vector $b \in \mathbb{R}^{d_{\text{out}}}$:

$$Y = X W + b$$

If `bias=False`:

$$Y = X W$$

---

## 💻 Ground-Up Implementation

```python
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
```

---

## 📊 Tensor Shapes & Parameters

| Parameter / Tensor | Symbol / Shape | Description |
|---|---|---|
| **Input Shape** | $(B, T, d_{\text{in}})$ | Batch size $B$, sequence length $T$, input dimension $d_{\text{in}}$ |
| **Output Shape** | $(B, T, d_{\text{out}})$ | Output projected tensor |
| **Weights Parameter** | $(d_{\text{in}}, d_{\text{out}})$ | Learnable weight parameters |
| **Bias Parameter** | $(d_{\text{out}})$ (optional) | Learnable bias vector |

### Parameter Formula

$$\text{Params} = \begin{cases} d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}} & \text{if bias is True} \\ d_{\text{in}} \cdot d_{\text{out}} & \text{if bias is False} \end{cases}$$
