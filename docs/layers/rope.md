# RotaryEmbedding (RoPE) Layer

Documentation for Rotary Position Embeddings (RoPE) implemented in `NNFS`.

---

## 💡 Overview

**Rotary Position Embedding (RoPE)** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) injects positional information into self-attention by rotating Query and Key vectors in complex 2D planes according to token sequence index $m$.

RoPE allows relative positional relationships to naturally naturally decay with distance and is used in architectures like **PaLM**, **LLaMA**, and **Mistral**.

Module Location: [`nnfs/layers/rope.py`](../../nnfs/layers/rope.py)

---

## 📐 Mathematical Formulation

For sequence index $m$ and 2D vector pair $(x_1, x_2)^T$:

$$R_{\Theta, m}^{(i)} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

where frequency $\theta_i = b^{-2(i-1)/d}$ with base $b = 10000.0$.

### Efficient Vectorized Rotation

$$\text{rotate\_half}(x) = [-x_{d/2:d}, x_{0:d/2}]$$
$$x_{\text{rope}} = (x \odot \cos) + (\text{rotate\_half}(x) \odot \sin)$$

---

## 💻 Ground-Up Implementation

```python
import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_position_embeddings:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
        return self.cos_cached[:, :, :seq_len, :].to(device), self.sin_cached[:, :, :seq_len, :].to(device)
```

---

## 📊 Tensor Shapes & Parameters

| Component | Shape | Description |
|---|---|---|
| **Query Tensor $q$** | $(B, n_{\text{heads}}, T, d_{\text{head}})$ | Multi-head query input |
| **Key Tensor $k$** | $(B, 1, T, d_{\text{head}})$ or $(B, n_{\text{heads}}, T, d_{\text{head}})$ | Key input tensor |
| **Cos / Sin Cache** | $(1, 1, T_{\text{max}}, d_{\text{head}})$ | Non-persistent buffer |
| **Learnable Parameters** | **0** | Calculated purely deterministically |
