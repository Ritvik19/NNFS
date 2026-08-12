# LLaMA 3 Architecture

Comprehensive documentation of the **LLaMA 3** architecture implemented in `NNFS`.

---

## 💡 Overview

LLaMA 3 is a family of state-of-the-art foundation models introduced by Meta AI in 2024 ([Grattafiori et al.](https://huggingface.co/papers/2407.21783)). LLaMA 3 spans sizes from 1B/3B (Llama 3.2 edge text models) and 8B/70B (Llama 3, 3.1, 3.3) to 405B (Llama 3.1 dense flagship model).

In `NNFS`, LLaMA 3 is implemented with modular primitives matching the original **RMSNorm Pre-LN**, **Grouped-Query Attention (GQA)**, **SwiGLU FFN**, and **Llama 3 Piecewise Frequency-Scaled Rotary Position Embeddings (RoPE)** with $\theta = 500,000.0$.

### Key Architectural Characteristics
- **Rotary Position Embeddings (RoPE) with Base $\theta = 500,000.0$**: Upgraded base frequency (from 10,000 in Llama 1/2 to 500,000 in Llama 3) to prevent phase collision and preserve angular distinctness across long sequences.
- **Piecewise Wavelength Frequency Scaling (`rope_scaling`)**: Applies Llama 3 piecewise frequency scaling (short wavelengths unscaled, long wavelengths scaled by context factor $S$, medium wavelengths smoothly interpolated) to support extended context windows up to 128K tokens.
- **Universal Grouped-Query Attention (GQA)**: Query heads ($n_{\text{heads}}$) are partitioned into groups sharing a smaller number of Key/Value heads ($n_{\text{kv\_heads}}$). Standardized across all model sizes ($N_{\text{kv\_heads}} = 8$).
- **Pre-normalization with RMSNorm**: Layer normalization is replaced with Root Mean Square Layer Normalization (`RMSNorm`), applied **before** attention and feed-forward sub-layers.
- **SwiGLU Activation Function**: Feed-forward expansion network utilizing SwiGLU gating ($\text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}})$) with $d_{\text{ff}} = \text{multiple\_of\_256}\left(\frac{2}{3} \cdot 4 d_{\text{model}}\right)$.
- **Bias-Free Linear Layers**: All linear projections ($W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$, and LM Head) omit bias parameters (`bias=False`).
- **Tied Embedding Weights**: Output classification head (`TiedLinear`) shares weights with the token embedding matrix (`Embedding`), matching parameter-efficient miniature implementation standards (and Llama 3.2 1B/3B).

---

## 🏗️ High-Level Architecture

The flow below details how input token sequences pass through RMSNorm Pre-LN blocks and final RMSNorm in LLaMA 3, with tensor dimensions using the baseline config (`vocab_size=256`, `d_model=256`, `block_size=1024`, `n_layers=4`, `n_heads=4`, `n_kv_heads=2`).

```mermaid
flowchart TD
    subgraph Input ["1. Input Pipeline"]
        TokenIDs["Input Token Indices<br/>Shape: (B, T)"]
        TokEmb["Token Embedding<br/>Shape: (B, T, 256)"]
        Drop["Embedding Dropout<br/>Shape: (B, T, 256)"]
        
        TokenIDs --> TokEmb
        TokEmb --> Drop
    end

    subgraph Blocks ["2. Transformer Backbone (4 x Llama3TransformerBlock)"]
        Drop --> Block1["LLaMA 3 Block 1<br/>Shape: (B, T, 256)"]
        Block1 --> Block2["LLaMA 3 Block 2<br/>Shape: (B, T, 256)"]
        Block2 --> Dots["..."]
        Dots --> Block4["LLaMA 3 Block 4<br/>Shape: (B, T, 256)"]
    end

    subgraph Head ["3. Output Normalization & Head"]
        Block4 --> RMSF["Final RMSNorm rms_f<br/>Shape: (B, T, 256)"]
        RMSF --> LMHead["TiedLinear Head<br/>Shape: (B, T, 256)"]
        LMHead --> Logits["Output Logits<br/>Shape: (B, T, 256)"]
    end
```

---

## 🧩 Transformer Block (Grouped-Query Attention & Llama 3 RoPE)

Each `Llama3TransformerBlock` applies RMSNorm prior to attention and SwiGLU operations, incorporating Grouped-Query Attention (GQA) with Llama 3 RoPE positional encoding ($\theta = 500,000.0$ and optional piecewise scaling).

```mermaid
flowchart TD
    In["Block Input x<br/>Shape: (B, T, 256)"] --> RMS1["RMSNorm 1<br/>Shape: (B, T, 256)"]
    RMS1 --> Attn["Grouped-Query Attention (Llama 3 RoPE)<br/>(4 Q heads, 2 KV heads, d_head=64, bias=False)<br/>Shape: (B, T, 256)"]
    In --> Add1["Residual Add (+)<br/>Shape: (B, T, 256)"]
    Attn --> Add1
    
    Add1 --> RMS2["RMSNorm 2<br/>Shape: (B, T, 256)"]
    RMS2 --> SwiGLU["SwiGLU FFN<br/>w_gate: 256 → 1024, w_up: 256 → 1024<br/>w_down: 1024 → 256 (bias=False)<br/>Shape: (B, T, 256)"]
    Add1 --> Add2["Residual Add (+)<br/>Shape: (B, T, 256)"]
    SwiGLU --> Add2
    
    Add2 --> Out["Block Output x_out<br/>Shape: (B, T, 256)"]
```

### Mathematical Formulations

Given block input $x \in \mathbb{R}^{B \times T \times d_{\text{model}}}$:

1. **Grouped-Query Attention Sub-layer**:
   $$\text{x}_{\text{norm1}} = \text{RMSNorm}_1(x)$$
   $$\text{x}_{\text{attn}} = \text{GroupedQueryAttention}_{\text{Llama3RoPE}}(\text{x}_{\text{norm1}})$$
   $$\text{x}_{\text{res1}} = x + \text{x}_{\text{attn}}$$

2. **SwiGLU Feed-Forward Sub-layer**:
   $$\text{x}_{\text{norm2}} = \text{RMSNorm}_2(\text{x}_{\text{res1}})$$
   $$\text{x}_{\text{ffn}} = \left(\text{Swish}(\text{x}_{\text{norm2}} W_{\text{gate}}) \odot (\text{x}_{\text{norm2}} W_{\text{up}})\right) W_{\text{down}}$$
   $$\text{x}_{\text{out}} = \text{x}_{\text{res1}} + \text{x}_{\text{ffn}}$$

3. **Final Normalization & Output Head**:
   $$\text{Logits} = \text{TiedLinear}\left(\text{RMSNorm}_f(\text{x}_{\text{final}})\right)$$

---

## ⚙️ Component Breakdown

### 1. `GroupedQueryAttention`
Projects Query vectors into $n_{\text{heads}}$ heads and Key/Value vectors into $n_{\text{kv\_heads}}$ heads. Key and Value projections are expanded across Query head groups using repeat interleave factor $N_{\text{rep}} = n_{\text{heads}} / n_{\text{kv\_heads}}$:
$$Q = \text{Linear}_q(x), \quad K = \text{Repeat}\left(\text{Linear}_k(x), N_{\text{rep}}\right), \quad V = \text{Repeat}\left(\text{Linear}_v(x), N_{\text{rep}}\right)$$
$$\text{Attention}(R_m Q, R_n K, V) = \text{Softmax}\left(\frac{(R_m Q) (R_n K)^T}{\sqrt{d_{\text{head}}}} + M\right) V$$

### 2. `RotaryEmbedding` (Llama 3 Piecewise Frequency Scaling)
Computes frequencies $\omega_i = \theta^{-2i/d}$ with base $\theta = 500,000$. For dimensions with wavelength $\lambda_i = 2\pi / \omega_i$:
- $\lambda_i < w_{\text{high}}$: Unscaled ($\omega_i' = \omega_i$).
- $\lambda_i > w_{\text{low}}$: Fully scaled by $S$ ($\omega_i' = \omega_i / S$).
- $w_{\text{high}} \le \lambda_i \le w_{\text{low}}$: Smoothly interpolated between $1.0$ and $1/S$.

### 3. `RMSNorm`
Normalizes inputs by scaling by the root mean square without mean centering:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

### 4. `SwiGLUMLP`
Feed-forward network utilizing SiLU (Swish) gated activations:
$$\text{SwiGLU}(x) = \left(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\right) W_{\text{down}}$$

### 5. `TiedLinear`
Reuses token embedding weights $W_{\text{tok}} \in \mathbb{R}^{V \times d_{\text{model}}}$ as transposed weights $W_{\text{head}} = W_{\text{tok}}^T \in \mathbb{R}^{d_{\text{model}} \times V}$ for language modeling output:
$$\text{Logits} = x \cdot W_{\text{tok}}^T$$

---

## 📊 Parameter & Shape Specifications

### Mini-LLaMA 3 Baseline Configurations (NNFS Default)

| Parameter | Symbol | Mini Value |
|---|---|---|
| Vocabulary Size | `vocab_size` | 256 |
| Context Length | `block_size` | 1024 |
| Hidden Dimension | `d_model` | 256 |
| Transformer Layers | `n_layers` | 4 |
| Query Attention Heads | `n_heads` | 4 |
| Key-Value Heads | `n_kv_heads` | 2 |
| Head Dimension | `d_head` | 64 |
| Feed-Forward Dim | `d_ff` | 1024 |
| RoPE Base Frequency | `rope_theta` | 500,000.0 |

### Parameter Breakdown

| Component | Parameters Formula | Count (Mini Config) |
|---|---|---|
| Token Embedding (`tok_embed`) | $V \times d_{\text{model}}$ | 65,536 |
| Positional Embedding (`pos_embed`) | None (RoPE based) | 0 |
| **Per Transformer Block ($\times 4$)** | | |
| - RMSNorms (`rms_1` + `rms_2`) | $2 \times d_{\text{model}}$ | 512 |
| - Attention (`q_proj` + `k_proj` + `v_proj` + `out_proj`, bias=False) | $d_{\text{model}}^2 + 2 \times (d_{\text{model}} \times n_{\text{kv\_heads}} \times d_{\text{head}}) + d_{\text{model}}^2$ | 196,608 |
| - SwiGLU FFN (`w_gate` + `w_up` + `w_down`) | $3 \times (d_{\text{model}} \times d_{\text{ff}})$ | 786,432 |
| **Total Block Params ($\times 4$)** | $4 \times 983,552$ | 3,934,208 |
| **Final RMSNorm (`rms_f`)** | $d_{\text{model}}$ | **256** |
| Final Head (`lm_head`) | Tied to `tok_embed` (0 new) | 0 |
| **Total Model Parameters** | | **4,000,000** |
