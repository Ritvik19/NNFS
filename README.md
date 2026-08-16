# NNFS

Neural Networks From Scratch

> Clean, modular, and readable miniature implementations of landmark deep learning models built ground-up.

---

## 📌 Overview

**NNFS** is a playground and repository dedicated to implementing miniature versions of key neural network architectures from fundamental components. Rather than relying on high-level pre-packaged model abstractions, NNFS builds models step-by-step from custom primitives—including layers, attention mechanisms, normalization, activations, and tokenizers.

---

## 🛠️ Repository Structure

```directory
nnfs/
├── configs/                   # Model and pipeline configuration files
├── nnfs/                      # Core neural network library from scratch
│   ├── activations/           # Custom activation functions (GELU, ReLU)
│   ├── layers/                # Basic layers (Linear, TiedLinear, Embedding, LayerNorm, Dropout, CausalMultiHeadAttention, MLP)
│   ├── losses/                # Loss functions (CrossEntropy)
│   ├── modules/               # Composite blocks (GPT1TransformerBlock, GPT2TransformerBlock)
│   ├── models/                # Complete model architectures (GPT-1, GPT-2)
│   ├── preprocessors/         # Tokenizers (CharTokenizer)
│   └── utils/                 # Causal LM training utilities, model I/O, generation
├── src/                       # Entry points for execution
│   ├── train.py               # Model training script
│   └── inference.py           # Text generation & sampling script
├── tests/                     # Comprehensive PyTest suite
├── run.sh                     # Quick launch shell script
└── README.md                  # Project documentation
```

---

## 🧩 Implemented Components & Models

### 🗂️ Models

- **GPT-1**: Post-layer-normalization decoder-only transformer with learned positional embeddings and tied weight output head.
- **GPT-2**: Pre-layer-normalization decoder-only transformer with final layer norm prior to the tied projection head.
- **PaLM**: Parallel decoder-only transformer with Multi-Query Attention (MQA), SwiGLU activations, Rotary Position Embeddings (RoPE), and bias-free kernels.
- **LLaMA 1**: Pre-RMSNorm decoder-only transformer with SwiGLU activations, Rotary Position Embeddings (RoPE), and bias-free projections.
- **LLaMA 2**: Pre-RMSNorm decoder-only transformer with Grouped-Query Attention (GQA), SwiGLU activations, Rotary Position Embeddings (RoPE), and bias-free projections.
- **LLaMA 3**: Pre-RMSNorm decoder-only transformer with Grouped-Query Attention (GQA), SwiGLU activations, Llama 3 piecewise frequency-scaled Rotary Position Embeddings ($\theta = 500,000.0$), and tied weight projections.
- **Transformer**: Modular decoder-only transformer with configurable positional encodings (sinusoidal, learned, ALiBi, RoPE, none), configurable activation functions (ReLU, GELU, SwiGLU), configurable normalization (LayerNorm, RMSNorm), and post-LN / pre-LN options.

### 📐 Layers & Modules ([Overview](./docs/layers/README.md))

- **[`RMSNorm`](./docs/layers/rms_norm.md)**: Root Mean Square Layer Normalization from Zhang & Sennrich (2019).
- **[`SinusoidalPositionalEncoding`](./docs/layers/sinusoidal_positional_encoding.md)**: Fixed sine/cosine positional encodings from Vaswani et al. (2017).
- **[`MultiQueryAttention`](./docs/layers/multi_query_attention.md)**: Multi-Query Attention with single shared Key/Value head and RoPE.
- **[`GroupedQueryAttention`](./docs/layers/grouped_query_attention.md)**: Grouped-Query Attention with configurable KV head groups and RoPE scaling.
- **[`SwiGLUMLP`](./docs/layers/swiglu_mlp.md)**: Feed-forward expansion network with SwiGLU gating activation.
- **[`RotaryEmbedding`](./docs/layers/rope.md)**: Rotary position embeddings (RoPE) applied to query and key projections with optional Llama 3 frequency scaling.
- **[`CausalMultiHeadAttention`](./docs/layers/causal_multi_head_attention.md)**: Scaled dot-product multi-head causal self-attention with masking.
- **[`Linear`](./docs/layers/linear.md) & [`TiedLinear`](./docs/layers/tied_linear.md)**: Weight-tied output classification head reusing token embedding weights.
- **[`LayerNorm`](./docs/layers/layer_norm.md)**: Standard layer normalization with learnable gain and bias.
- **[`MLP`](./docs/layers/mlp.md)**: Feed-forward expansion network (`d_model` $\rightarrow$ `d_ff` $\rightarrow$ `d_model`) with GELU or ReLU activation.
- **[`Embedding`](./docs/layers/embedding.md)**: Lookup embedding matrix for input tokens and positions.
- **[`Dropout`](./docs/layers/dropout.md)**: Inverted dropout for regularization during training.

### ⚡ Activations ([Overview](./docs/activations/README.md))

- **[`ReLU`](./docs/activations/relu.md)**: Piecewise linear non-linearity ($\max(0, x)$).
- **[`GELU`](./docs/activations/gelu.md)**: Gaussian Error Linear Unit activation with fast tanh approximation.
- **[`SwiGLU`](./docs/activations/swiglu.md)**: Swish-Gated Linear Unit ($\text{Swish}_{\beta}(\text{gate}) \odot \text{up}$).

### 🔤 Preprocessing

- **`CharTokenizer`**: Character-level vocabulary encoder/decoder with special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).

### 🗗 Model Training Logs

| Model                        | Architecture                                 | # Parameters          | Training Log                                                           | Train Loss | Eval Loss |
| ---------------------------- | -------------------------------------------- | --------------------- | ---------------------------------------------------------------------- | ---------- | --------- |
| Transformer                  | [Architecture](./docs/models/transformer.md) | 3,224,576             | [link](https://wandb.ai/ritvik19/nnfs/runs/stugg5kc?nw=nwuserritvik19) | 0.57989    | 0.50835   |
| Transformer - GELU           | -                                            | 3,224,576             | [link](https://wandb.ai/ritvik19/nnfs/runs/4vlaylw0?nw=nwuserritvik19) | 0.50243    | 0.48441   |
| Transformer - SwiGLU         | -                                            | 4,268,032             | [link](https://wandb.ai/ritvik19/nnfs/runs/23io8gg9?nw=nwuserritvik19) | 0.50665    | 0.47742   |
| Transformer - Learned PE     | -                                            | 3,486,720             | [link](https://wandb.ai/ritvik19/nnfs/runs/o4jjrvkx?nw=nwuserritvik19) | 0.5911     | 0.52374   |
| Transformer - ALiBi          | -                                            | 3,224,576             | [link](https://wandb.ai/ritvik19/nnfs/runs/gsk5gty0?nw=nwuserritvik19) | 0.51397    | 0.46871   |
| Transformer - RoPE           | -                                            | 3,224,576             | [link](https://wandb.ai/ritvik19/nnfs/runs/stfws7au?nw=nwuserritvik19) | 0.50446    | 0.47498   |
| Transformer - RMSNorm        | -                                            | 3,224,576             | [link](https://wandb.ai/ritvik19/nnfs/runs/hcfuzjx1?nw=nwuserritvik19) | 0.65418    | 0.53732   |
| Transformer - MQA            | -                                            | 2,961,408             | [link](https://wandb.ai/ritvik19/nnfs/runs/dl4bi38t?nw=nwuserritvik19) | 0.69114    | 0.57979   |
| Transformer - GQA            | -                                            | 2,961,408             | [link](https://wandb.ai/ritvik19/nnfs/runs/chbx0ofp?nw=nwuserritvik19) | 0.56067    | 0.51578   |
| GPT1                         | [Architecture](./docs/models/gpt1.md)        | 3,486,720             | [link](https://wandb.ai/ritvik19/nnfs/runs/fhjgffh9?nw=nwuserritvik19) | 0.63432    | 0.51363   |
| GPT2                         | [Architecture](./docs/models/gpt2.md)        | 3,487,232             | [link](https://wandb.ai/ritvik19/nnfs/runs/rdr6rk7j?nw=nwuserritvik19) | 0.56123    | 0.47952   |
| PaLM                         | [Architecture](./docs/models/palm.md)        | 3,869,184             | [Link](https://wandb.ai/ritvik19/nnfs/runs/ojgu35k6?nw=nwuserritvik19) | 0.42648    | 0.42843   |
| PaLM 2                       | [Architecture](./docs/models/palm2.md)       | 3,998,976             | [Link](https://wandb.ai/ritvik19/nnfs/runs/plj8t4k0?nw=nwuserritvik19) | 0.41657    | 0.42777   |
| Llama 1                      | [Architecture](./docs/models/llama1.md)      | 4,262,144             | [Link](https://wandb.ai/ritvik19/nnfs/runs/dkzq9y5e?nw=nwuserritvik19) | 0.40327    | 0.37973   |
| Llama 2                      | [Architecture](./docs/models/llama2.md)      | 4,000,000             | [Link](https://wandb.ai/ritvik19/nnfs/runs/tkw5quka?nw=nwuserritvik19) | 0.47521    | 0.38264   |
| Llama 3                      | [Architecture](./docs/models/llama3.md)      | 4,000,000             | -                                                                      | -          | -         |
| Mistral                      | [Architecture](./docs/models/mistral.md)     | 4,000,000             | [Link](https://wandb.ai/ritvik19/nnfs/runs/rkegoz2o?nw=nwuserritvik19) | 0.46780    | 0.39532   |
| Mistral - Interleaved SWA    | -                                            | 4,000,000             | [Link](https://wandb.ai/ritvik19/nnfs/runs/fvqrqysu?nw=nwuserritvik19) | 0.38006    | 0.38085   |
| Mixtral MOE                  | [Architecture](./docs/models/mixtral_moe.md) | 4,008,192 / 1,648,896 | [Link](https://wandb.ai/ritvik19/nnfs/runs/60jeu7fc?nw=nwuserritvik19) | 0.45857    | 0.39268   |
| Mixtral MOE + Load Balancing | -                                            | -                     | [Link](https://wandb.ai/ritvik19/nnfs/runs/60jeu7fc?nw=nwuserritvik19) | 0.44111    | 0.38565   |
| GPT OSS MOE                  | [Architecture](./docs/models/gpt_oss.md)     | 4,011,280 / 1,717,520 | [Link](https://wandb.ai/ritvik19/nnfs/runs/uoq0cu05?nw=nwuserritvik19) | 0.4395     | 0.3852    |

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/nnfs.git
cd nnfs

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or: pip install torch pyyaml wandb datasets pytest
```

### 2. Training a Model

To launch training using configured YAML parameters:

```bash
python src/train.py --model-config configs/gpt2_config.yaml --train-config configs/train_config.yaml
```

_Or run via the quick script:_

```bash
bash run.sh
```

### 3. Running Inference / Text Generation

Generate text from a trained checkpoint or initialized model:

```bash
python src/inference.py --inference-config configs/inference_config.yaml
```
