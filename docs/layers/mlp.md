# MLP (Multi-Layer Perceptron) Layer

Documentation for the standard `MLP` feed-forward layer implemented in `NNFS`.

---

## 💡 Overview

The `MLP` layer is a two-layer feed-forward expansion network used within Transformer blocks (such as in **GPT-1** and **GPT-2**). It expands token hidden representations from $d_{\text{model}}$ to $d_{\text{ff}}$ (typically $4 \times d_{\text{model}}$), applies non-linear activation (e.g. GELU), and projects back to $d_{\text{model}}$.

Module Location: [`nnfs/layers/mlp.py`](../../nnfs/layers/mlp.py)

---

## 🏗️ Structure Flow

```mermaid
flowchart LR
    In["Input (B, T, d_model)"] --> FC1["Linear fc1 (d_model -> d_ff)"]
    FC1 --> Act["Activation (GELU / ReLU)"]
    Act --> Drop["Dropout"]
    Drop --> FC2["Linear fc2 (d_ff -> d_model)"]
    FC2 --> Out["Output (B, T, d_model)"]
```

---

## 📐 Mathematical Formulation

Given input representation $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$:

$$H_1 = \text{Activation}\left(X W_1 + b_1\right)$$
$$H_2 = \text{Dropout}\left(H_1\right)$$
$$Y = H_2 W_2 + b_2$$

where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ and $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$.

---

## 💻 Ground-Up Implementation

```python
import torch
import torch.nn as nn
from .dropout import Dropout
from .linear import Linear

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = Dropout(dropout)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(input))))
```

---

## 📊 Tensor Shapes & Parameters

| Component | Shape | Parameter Formula |
|---|---|---|
| **Input Linear Layer (`fc1`)** | Weight: $(d_{\text{model}}, d_{\text{ff}})$, Bias: $(d_{\text{ff}})$ | $d_{\text{model}} \cdot d_{\text{ff}} + d_{\text{ff}}$ |
| **Output Linear Layer (`fc2`)** | Weight: $(d_{\text{ff}}, d_{\text{model}})$, Bias: $(d_{\text{model}})$ | $d_{\text{ff}} \cdot d_{\text{model}} + d_{\text{model}}$ |
| **Total Parameters** | | **$2 d_{\text{model}} d_{\text{ff}} + d_{\text{ff}} + d_{\text{model}}$** |
