# Embedding Layer

Documentation for the `Embedding` lookup table layer implemented in `NNFS`.

---

## 💡 Overview

The `Embedding` layer maps discrete token indices or position indices into dense vector representations. In `NNFS`, `Embedding` is implemented ground-up using a raw `nn.Parameter` matrix and direct tensor index slicing (`self.embed[input]`).

Module Location: [`nnfs/layers/embedding.py`](../../nnfs/layers/embedding.py)

---

## 📐 Mathematical Formulation

Given vocabulary size $V$, embedding dimension $d_{\text{model}}$, parameter table $E \in \mathbb{R}^{V \times d_{\text{model}}}$, and index tensor $I \in \mathbb{Z}^{B \times T}$:

$$Y_{b, t} = E_{I_{b, t}} \in \mathbb{R}^{d_{\text{model}}}$$

---

## 💻 Ground-Up Implementation

```python
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(Embedding, self).__init__()
        self.embed = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embed[input]
```

---

## 📊 Tensor Shapes & Parameters

| Parameter / Tensor | Symbol / Shape | Description |
|---|---|---|
| **Input Shape** | $(B, T)$ | Tensor of integer indices in range $[0, V-1]$ |
| **Output Shape** | $(B, T, d_{\text{model}})$ | Dense embedding vectors |
| **Embedding Table** | $(V, d_{\text{model}})$ | Learnable representation matrix |

### Parameter Formula

$$\text{Params} = V \cdot d_{\text{model}}$$
