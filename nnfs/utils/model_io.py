import os
from typing import Union

import torch
import yaml

from nnfs.models import (
    GPT1,
    GPT2,
    Llama1,
    PaLM,
    Transformer,
    GPT1Config,
    GPT2Config,
    Llama1Config,
    PaLMConfig,
    TransformerConfig,
)
from nnfs.preprocessors.char_tokenizer import CharTokenizer

Tokenizer = Union[CharTokenizer]
Model = Union[GPT1, GPT2, PaLM, Transformer, Llama1]
Config = Union[GPT1Config, GPT2Config, PaLMConfig, TransformerConfig, Llama1Config]

MODEL_REGISTRY = {
    "gpt1": (GPT1, GPT1Config),
    "gpt2": (GPT2, GPT2Config),
    "palm": (PaLM, PaLMConfig),
    "transformer": (Transformer, TransformerConfig),
    "llama1": (Llama1, Llama1Config),
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
