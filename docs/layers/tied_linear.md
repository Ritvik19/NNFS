# TiedLinear Layer

Documentation for the `TiedLinear` weight-tied classification layer implemented in `NNFS`.

---

## 💡 Overview

`TiedLinear` is a subclass of `Linear` designed to implement **Weight Tying** between token embedding matrices and language model output projection heads (as described in [Press & Wolf, 2017](https://arxiv.org/abs/1608.05859)).

Rather than instantiating a separate projection weight matrix of size $d_{\text{model}} \times V$, `TiedLinear` dynamically reuses the transpose of the `Embedding` parameter matrix ($W_{\text{emb}}^T \in \mathbb{R}^{d_{\text{model}} \times V}$).

Module Location: [`nnfs/layers/tied_linear.py`](../../nnfs/layers/tied_linear.py)

---

## 📐 Mathematical Formulation

Given hidden state representation matrix $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ and token embedding parameter $E \in \mathbb{R}^{V \times d_{\text{model}}}$:

$$Y = X E^T + b$$

where $E^T \in \mathbb{R}^{d_{\text{model}} \times V}$ and optional bias $b \in \mathbb{R}^V$.

---

## 💻 Ground-Up Implementation

```python
import torch
from .linear import Linear
from .embedding import Embedding

class TiedLinear(Linear):
    def __init__(self, embedding: Embedding, bias: bool = True):
        super().__init__(embedding.embed.shape[1], embedding.embed.shape[0], bias=bias)
        del self._parameters["weights"]
        self.embedding = embedding

    @property
    def weights(self) -> torch.Tensor:
        return self.embedding.embed.t()
```

---

## 📊 Tensor Shapes & Parameters

| Attribute | Specification |
|---|---|
| **Input Shape** | $(B, T, d_{\text{model}})$ |
| **Output Shape** | $(B, T, \text{vocab\_size})$ |
| **New Weight Parameters** | **0** (Shared with `Embedding` matrix of size $V \times d_{\text{model}}$) |
| **Bias Parameter** | $V$ parameters (if `bias=True`) |

---

## ⚙️ Key Advantages

1. **Parameter Reduction**: Eliminates millions of unnecessary parameters in the output linear head.
2. **Regularization & Representation**: Synchronizes input token representation space with output logit scoring.
3. **Used Across Models**: Applied in **GPT-1**, **GPT-2**, and **PaLM** heads in `NNFS`.
