import os
from typing import Union

import torch
import yaml

from nnfs.models import (
    GPT1,
    GPT2,
    Llama1,
    Llama2,
    Llama3,
    Mistral,
    MixtralMoE,
    PaLM,
    PaLM2,
    Transformer,
    GPT1Config,
    GPT2Config,
    Llama1Config,
    Llama2Config,
    Llama3Config,
    MistralConfig,
    MixtralMoEConfig,
    PaLMConfig,
    PaLM2Config,
    TransformerConfig,
)
from nnfs.preprocessors.char_tokenizer import CharTokenizer

Tokenizer = Union[CharTokenizer]
Model = Union[GPT1, GPT2, PaLM, PaLM2, Transformer, Llama1, Llama2, Llama3, Mistral, MixtralMoE]
Config = Union[
    GPT1Config,
    GPT2Config,
    PaLMConfig,
    PaLM2Config,
    TransformerConfig,
    Llama1Config,
    Llama2Config,
    Llama3Config,
    MistralConfig,
    MixtralMoEConfig,
]

MODEL_REGISTRY = {
    "gpt1": (GPT1, GPT1Config),
    "gpt2": (GPT2, GPT2Config),
    "palm": (PaLM, PaLMConfig),
    "palm2": (PaLM2, PaLM2Config),
    "transformer": (Transformer, TransformerConfig),
    "llama1": (Llama1, Llama1Config),
    "llama2": (Llama2, Llama2Config),
    "llama3": (Llama3, Llama3Config),
    "mistral": (Mistral, MistralConfig),
    "mixtral_moe": (MixtralMoE, MixtralMoEConfig),
}


def build_model(config_file_path: str):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    model_name = config["model_name"]
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_name {model_name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    model_class, model_config = MODEL_REGISTRY[model_name]
    del config["model_name"]
    model = model_class(model_config(**config))
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    if model_name.endswith("_moe") or hasattr(model, "count_active_parameters"):
        if hasattr(model, "count_active_parameters"):
            active_params = model.count_active_parameters()
        else:
            active_params = num_params
        print(f"Active parameters per token: {active_params:,}")

    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,}")

    return model


def load_model(
    model_path: str,
    model_name: str = "gpt1",
    device: torch.device | str | None = None,
) -> Model:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_name {model_name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    model_class, _ = MODEL_REGISTRY[model_name]
    config = torch.load(
        os.path.join(model_path, "config.pth"),
        map_location=device,
        weights_only=False,
    )
    model = model_class(config)
    model.load_pretrained(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model
