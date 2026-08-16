import argparse
import logging
import os
import sys

import torch
import yaml

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nnfs.preprocessors.char_tokenizer import CharTokenizer
from nnfs.utils.model_io import load_model
from nnfs.utils.text_generation import generate_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference")


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained GPT-1 model")
    parser.add_argument(
        "--inference-config",
        type=str,
        help="Path to inference config YAML file",
    )
    args = parser.parse_args()

    inference_config = {}
    if os.path.exists(args.inference_config):
        with open(args.inference_config, "r", encoding="utf-8") as f:
            inference_config = yaml.safe_load(f) or {}
    else:
        raise FileNotFoundError(f"Inference config not found: {args.inference_config}")

    model_name = inference_config.get("model_name", None)
    model_path = inference_config.get("model_path", "checkpoints")
    temperature = inference_config.get("temperature", 1.0)
    top_p = inference_config.get("top_p")
    top_k = inference_config.get("top_k")
    max_length = inference_config.get("max_length", 100)
    prompts = inference_config.get("prompts", [])

    if not prompts:
        raise ValueError("No prompts provided. Set prompts in the config.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference config: %s", inference_config)
    logger.info("Device: %s | model: %s | path: %s", device, model_name, model_path)

    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = CharTokenizer.load(tokenizer_path)
    logger.info("Tokenizer vocab size: %d", tokenizer.vocab_size)

    model = load_model(model_path, model_name=model_name, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded model | params: %s | block_size: %d | vocab_size: %d",
        f"{num_params:,}",
        model.config.block_size,
        model.config.vocab_size,
    )

    for i, prompt in enumerate(prompts, start=1):
        logger.info("Generating [%d/%d] for prompt: %r", i, len(prompts), prompt)
        generated = generate_text(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        print(f"\n=== Prompt {i} ===\n{prompt}\n\n=== Generation ===\n{generated}\n")


if __name__ == "__main__":
    main()
