# Dropout Layer

Documentation for the `Dropout` regularization layer implemented in `NNFS`.

---

## 💡 Overview

**Dropout** ([Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)) randomly zeroes elements of the input tensor during training with probability $p$. In `NNFS`, `Dropout` is implemented using **Inverted Dropout**, scaling surviving activations by $\frac{1}{1 - p}$ during training so that evaluation requires no modification.

Module Location: [`nnfs/layers/dropout.py`](../../nnfs/layers/dropout.py)

---

## 📐 Mathematical Formulation

During training (`self.training == True` and $p > 0$):

$$y = \frac{x \odot M}{1 - p}$$

where $M_{i} \sim \text{Bernoulli}(1 - p)$ is a random binary mask tensor ($M_{i} = 1$ with probability $1 - p$).

During evaluation (`self.training == False` or $p = 0$):

$$y = x$$

---

## 💻 Ground-Up Implementation

```python
import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return input
        keep = torch.rand_like(input) >= self.p
        return input * keep / (1.0 - self.p)
```

---

## 📊 Tensor Shapes & Parameters

| Attribute | Specification |
|---|---|
| **Input Shape** | $(B, \dots, D)$ |
| **Output Shape** | $(B, \dots, D)$ |
| **Dropout Probability ($p$)** | `float` (default $0.1$) |
| **Learnable Parameters** | **0** |
