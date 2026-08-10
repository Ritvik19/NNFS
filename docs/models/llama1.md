# LLaMA 1 Architecture & Miniature Implementation

Documentation for the `Llama1` decoder-only transformer model implemented in `NNFS`.

---

## 💡 Overview

**LLaMA 1** ([Touvron et al., Meta, 2023](https://arxiv.org/abs/2302.13971)) is an open and efficient collection of foundation language models ranging from 7B to 65B parameters. `NNFS` provides a clean, modular miniature implementation of LLaMA 1 built ground-up from its foundational components.

Module Location: [`nnfs/models/llama1.py`](../../nnfs/models/llama1.py)

---

## 📐 Key Architectural Components

1. **Pre-normalization with RMSNorm**: Input to every attention block and feed-forward block is normalized using `RMSNorm` (Root Mean Square Layer Normalization) instead of standard LayerNorm.
2. **SwiGLU Activation Function**: Replaces standard ReLU/GELU activations in the feed-forward network with SwiGLU gating ($\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}})$). Hidden dimension is scaled as $d_{\text{ff}} = \text{multiple\_of\_256}(\frac{8}{3} d_{\text{model}})$.
3. **Rotary Position Embeddings (RoPE)**: Removes absolute positional embeddings and applies RoPE to Query and Key projections in every attention layer.
4. **Bias-Free Linear Projections**: All dense projections ($W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$, and LM Head) omit bias parameters (`bias=False`).

---

## 💻 Ground-Up Implementation

```python
import torch
import torch.nn as nn
from nnfs.layers import Embedding, RMSNorm, TiedLinear, Dropout
from nnfs.modules import Llama1TransformerBlock

class Llama1Config:
    def __init__(
        self,
        vocab_size: int = 32000,
        block_size: int = 512,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.0,
        eps: float = 1e-5,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        if d_ff is None:
            raw_d_ff = int(2 * 4 * d_model / 3)
            self.d_ff = ((raw_d_ff + 255) // 256) * 256
        else:
            self.d_ff = d_ff
        self.dropout = dropout
        self.eps = eps
```

---

## 📊 Model Hyperparameters Across Official Sizes

| Model Size | $d_{\text{model}}$ | $n_{\text{heads}}$ | $n_{\text{layers}}$ | $d_{\text{ff}}$ (SwiGLU) |
|---|---|---|---|---|
| **7B** | 4096 | 32 | 32 | 11008 |
| **13B** | 5120 | 40 | 40 | 13824 |
| **33B** | 6656 | 52 | 60 | 17920 |
| **65B** | 8192 | 64 | 80 | 22016 |
| **Miniature (NNFS Default)** | 512 | 8 | 6 | 1536 |
