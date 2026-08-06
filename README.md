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
* **GPT-1**: Post-layer-normalization decoder-only transformer with learned positional embeddings and tied weight output head.
* **GPT-2**: Pre-layer-normalization decoder-only transformer with final layer norm prior to the tied projection head.
* **PaLM**: Parallel decoder-only transformer with Multi-Query Attention (MQA), SwiGLU activations, Rotary Position Embeddings (RoPE), and bias-free kernels.

### 📐 Layers & Modules ([Overview](./docs/layers/README.md))
* **[`MultiQueryAttention`](./docs/layers/multi_query_attention.md)**: Multi-Query Attention with single shared Key/Value head and RoPE.
* **[`SwiGLUMLP`](./docs/layers/swiglu_mlp.md)**: Feed-forward expansion network with SwiGLU gating activation.
* **[`RotaryEmbedding`](./docs/layers/rope.md)**: Rotary position embeddings (RoPE) applied to query and key projections.
* **[`CausalMultiHeadAttention`](./docs/layers/causal_multi_head_attention.md)**: Scaled dot-product multi-head causal self-attention with masking.
* **[`Linear`](./docs/layers/linear.md) & [`TiedLinear`](./docs/layers/tied_linear.md)**: Weight-tied output classification head reusing token embedding weights.
* **[`LayerNorm`](./docs/layers/layer_norm.md)**: Standard layer normalization with learnable gain and bias.
* **[`MLP`](./docs/layers/mlp.md)**: Feed-forward expansion network (`d_model` $\rightarrow$ `d_ff` $\rightarrow$ `d_model`) with GELU activation.
* **[`Embedding`](./docs/layers/embedding.md)**: Lookup embedding matrix for input tokens and positions.
* **[`Dropout`](./docs/layers/dropout.md)**: Inverted dropout for regularization during training.

### ⚡ Activations ([Overview](./docs/activations/README.md))
* **[`ReLU`](./docs/activations/relu.md)**: Piecewise linear non-linearity ($\max(0, x)$).
* **[`GELU`](./docs/activations/gelu.md)**: Gaussian Error Linear Unit activation with fast tanh approximation.
* **[`SwiGLU`](./docs/activations/swiglu.md)**: Swish-Gated Linear Unit ($\text{Swish}_{\beta}(\text{gate}) \odot \text{up}$).

### 🔤 Preprocessing
* **`CharTokenizer`**: Character-level vocabulary encoder/decoder with special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).

### 🗗 Model Training Logs

| Model | Architecture | # Parameters | Training Log | Eval Loss |
|-------|--------------|--------------|--------------|-----------|
| GPT1  | [Architecture](./docs/models/gpt1.md) | 3,486,720    | [link](https://wandb.ai/ritvik19/nnfs/runs/fhjgffh9?nw=nwuserritvik19) | 0.51363   |
| GPT2  | [Architecture](./docs/models  /gpt2.md) | 3,487,232    | [link](https://wandb.ai/ritvik19/nnfs/runs/rdr6rk7j?nw=nwuserritvik19) | 0.47952   |
| PaLM  | [Architecture](./docs/models/palm.md) | 3,869,184    | [Link](https://wandb.ai/ritvik19/nnfs/runs/ojgu35k6?nw=nwuserritvik19) | 0.42843 |


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

*Or run via the quick script:*
```bash
bash run.sh
```

### 3. Running Inference / Text Generation

Generate text from a trained checkpoint or initialized model:

```bash
python src/inference.py --inference-config configs/inference_config.yaml
```
