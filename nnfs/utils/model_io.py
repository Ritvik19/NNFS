import os
from typing import Union

import torch
import yaml

from nnfs.models import (
    GPT1,
    GPT2,
    GptOss,
    Llama1,
    Llama2,
    Llama3,
    Llama4,
    Mistral,
    MixtralMoE,
    PaLM,
    PaLM2,
    Transformer,
    GPT1Config,
    GPT2Config,
    GptOssConfig,
    Llama1Config,
    Llama2Config,
    Llama3Config,
    Llama4Config,
    MistralConfig,
    MixtralMoEConfig,
    PaLMConfig,
    PaLM2Config,
    TransformerConfig,
)
from nnfs.preprocessors.char_tokenizer import CharTokenizer

Tokenizer = Union[CharTokenizer]
Model = Union[GPT1, GPT2, PaLM, PaLM2, Transformer, Llama1, Llama2, Llama3, Llama4, Mistral, MixtralMoE, GptOss]
Config = Union[
    GPT1Config,
    GPT2Config,
    PaLMConfig,
    PaLM2Config,
    TransformerConfig,
    Llama1Config,
    Llama2Config,
    Llama3Config,
    Llama4Config,
    MistralConfig,
    MixtralMoEConfig,
    GptOssConfig,
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
    "llama4": (Llama4, Llama4Config),
    "llama_4": (Llama4, Llama4Config),
    "llama4_moe": (Llama4, Llama4Config),
    "llama_4_moe": (Llama4, Llama4Config),
    "mistral": (Mistral, MistralConfig),
    "mixtral_moe": (MixtralMoE, MixtralMoEConfig),
    "gpt_oss_moe": (GptOss, GptOssConfig),
    "gpt_oss": (GptOss, GptOssConfig),
}


CONFIG_REGISTRY = {
    GPT1Config: GPT1,
    GPT2Config: GPT2,
    PaLMConfig: PaLM,
    PaLM2Config: PaLM2,
    TransformerConfig: Transformer,
    Llama1Config: Llama1,
    Llama2Config: Llama2,
    Llama3Config: Llama3,
    Llama4Config: Llama4,
    MistralConfig: Mistral,
    MixtralMoEConfig: MixtralMoE,
    GptOssConfig: GptOss,
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
    model_name: str | None = None,
    device: torch.device | str | None = None,
) -> Model:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    config_path = os.path.join(model_path, "config.pth")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = torch.load(
        config_path,
        map_location=device,
        weights_only=False,
    )

    if model_name is not None:
        if model_name not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown model_name {model_name!r}. Available: {sorted(MODEL_REGISTRY)}"
            )
        model_class, _ = MODEL_REGISTRY[model_name]
    else:
        config_cls = type(config)
        if config_cls in CONFIG_REGISTRY:
            model_class = CONFIG_REGISTRY[config_cls]
        else:
            matched_cls = None
            for reg_cfg_cls, reg_mdl_cls in CONFIG_REGISTRY.items():
                if reg_cfg_cls.__name__ == config_cls.__name__:
                    matched_cls = reg_mdl_cls
                    break
            if matched_cls is not None:
                model_class = matched_cls
            else:
                raise ValueError(
                    f"Could not infer model architecture from config type {config_cls.__name__!r}. "
                    f"Available configs: {[c.__name__ for c in CONFIG_REGISTRY]}. "
                    f"Please specify `model_name` explicitly."
                )

    model = model_class(config)
    model.load_pretrained(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model
