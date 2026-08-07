# SinusoidalPositionalEncoding Layer

Comprehensive documentation for the `SinusoidalPositionalEncoding` primitive layer implemented in `NNFS`.

---

## 💡 Overview

`SinusoidalPositionalEncoding` computes non-trainable, fixed positional encodings using alternating sine and cosine functions as introduced in Section 3.5 of *"Attention Is All You Need"* ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

Because attention mechanisms possess no inherent order awareness, positional encodings are added to input token embeddings to provide the model with relative and absolute position information.

---

## 📐 Mathematical Formulation

For position $pos \in [0, \text{max\_len}-1]$ and dimension index $i \in [0, d_{\text{model}}/2 - 1]$:

$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### Key Properties
1. **Wavelength Spectrum**: Wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.
2. **Linear Translation Property**: For any fixed offset $k$, $\text{PE}_{pos+k}$ can be represented as a linear function of $\text{PE}_{pos}$, allowing the model to attend easily by relative positions.
3. **Parameter Efficiency**: Fixed buffer with zero trainable parameters (`register_buffer`).

---

## ⚙️ Input / Output Specifications

- **Parameters**:
  - `max_len` (`int`): Maximum sequence length to precompute.
  - `d_model` (`int`): Model hidden dimension (must be even).
- **Input**:
  - Sequence length `T` (`int`), or position index tensor `(T,)`, or input tensor `(B, T, C)`.
- **Output**:
  - Positional encoding tensor of shape `(T, d_model)` or `(1, T, d_model)`.

---

## 💻 Code Example

```python
import torch
from nnfs.layers import SinusoidalPositionalEncoding

pe_layer = SinusoidalPositionalEncoding(max_len=512, d_model=512)

# Retrieve encodings for sequence length 64
pe = pe_layer(64)  # Shape: (64, 512)
```
