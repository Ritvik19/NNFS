import argparse
import json
import logging
import os
import random
import sys

import torch
import torch.nn as nn

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nnfs.preprocessors.char_tokenizer import CharTokenizer
from nnfs.utils.causal_language_modelling import (
    CausalLanguageModelingDataLoader,
    CausalLanguageModelingDataset,
)
from nnfs.utils.model_io import build_model, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("router_analysis")


class RouterTracker:
    """Attaches PyTorch forward hooks to TopKRouter modules in MoE models to record token routing metrics."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.router_data = {}
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "TopKRouter":
                num_experts = getattr(module, "num_experts", 8)
                self.router_data[name] = {
                    "num_experts": num_experts,
                    "counts": torch.zeros(num_experts, dtype=torch.long),
                    "weights": torch.zeros(num_experts, dtype=torch.float64),
                    "total_token_slots": 0,
                }
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                weights, indices = output[0], output[1]
                indices_flat = indices.detach().reshape(-1)
                weights_flat = weights.detach().reshape(-1).to(torch.float64)

                data = self.router_data[layer_name]
                num_experts = data["num_experts"]

                for exp_idx in range(num_experts):
                    mask = indices_flat == exp_idx
                    count = mask.sum().item()
                    if count > 0:
                        data["counts"][exp_idx] += count
                        data["weights"][exp_idx] += weights_flat[mask].sum().item()

                data["total_token_slots"] += indices_flat.numel()

        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def compute_metrics(self) -> dict:
        results = {}
        global_counts = None
        global_weights = None
        global_slots = 0

        for layer_name, data in self.router_data.items():
            counts = data["counts"]
            weights = data["weights"]
            total_slots = data["total_token_slots"]
            num_experts = data["num_experts"]

            if global_counts is None:
                global_counts = torch.zeros(num_experts, dtype=torch.long)
                global_weights = torch.zeros(num_experts, dtype=torch.float64)

            global_counts += counts
            global_weights += weights
            global_slots += total_slots

            percentages = (counts.double() / max(total_slots, 1) * 100.0).tolist()
            avg_weights = [
                (weights[i].item() / max(counts[i].item(), 1)) if counts[i] > 0 else 0.0
                for i in range(num_experts)
            ]

            counts_float = counts.float()
            mean_c = counts_float.mean().item()
            std_c = counts_float.std().item() if num_experts > 1 else 0.0
            cv = (std_c / mean_c) if mean_c > 0 else 0.0

            results[layer_name] = {
                "num_experts": num_experts,
                "total_token_slots": total_slots,
                "counts": counts.tolist(),
                "percentages": [round(p, 2) for p in percentages],
                "avg_weights": [round(w, 4) for w in avg_weights],
                "cv_imbalance": round(cv, 4),
            }

        if global_counts is not None and global_slots > 0:
            num_experts = global_counts.size(0)
            global_pct = (global_counts.double() / global_slots * 100.0).tolist()
            global_avg_w = [
                (global_weights[i].item() / max(global_counts[i].item(), 1))
                if global_counts[i] > 0
                else 0.0
                for i in range(num_experts)
            ]
            counts_float = global_counts.float()
            mean_c = counts_float.mean().item()
            std_c = counts_float.std().item() if num_experts > 1 else 0.0
            cv = (std_c / mean_c) if mean_c > 0 else 0.0

            results["global_summary"] = {
                "total_token_slots": global_slots,
                "counts": global_counts.tolist(),
                "percentages": [round(p, 2) for p in global_pct],
                "avg_weights": [round(w, 4) for w in global_avg_w],
                "cv_imbalance": round(cv, 4),
            }

        return results


def print_analysis_report(metrics: dict) -> None:
    print("\n" + "=" * 80)
    print("                      ROUTER EXPERT ANALYSIS REPORT                      ")
    print("=" * 80)

    for layer_name, data in metrics.items():
        if layer_name == "global_summary":
            continue
        print(f"\n[ Layer: {layer_name} ]")
        print(f"Total Token Routing Slots: {data['total_token_slots']:,}")
        print(f"Load Imbalance Coefficient of Variation (CV): {data['cv_imbalance']}")

        header = f"{'Expert ID':<12} | {'Token Count':<14} | {'Routing Load %':<16} | {'Avg Routing Weight':<18}"
        divider = "-" * len(header)
        print(divider)
        print(header)
        print(divider)

        for exp_idx, (cnt, pct, weight) in enumerate(
            zip(data["counts"], data["percentages"], data["avg_weights"])
        ):
            print(f"Expert {exp_idx:<5} | {cnt:<14,} | {pct:<15.2f}% | {weight:<18.4f}")
        print(divider)

    if "global_summary" in metrics:
        g = metrics["global_summary"]
        print("\n" + "=" * 80)
        print("                     GLOBAL SUMMARY (ALL MOE LAYERS)                     ")
        print("=" * 80)
        print(f"Total Model-Wide Token Slots: {g['total_token_slots']:,}")
        print(f"Global Load Imbalance (CV): {g['cv_imbalance']}")

        header = f"{'Expert ID':<12} | {'Total Tokens':<14} | {'Overall Load %':<16} | {'Avg Routing Weight':<18}"
        divider = "-" * len(header)
        print(divider)
        print(header)
        print(divider)

        for exp_idx, (cnt, pct, weight) in enumerate(
            zip(g["counts"], g["percentages"], g["avg_weights"])
        ):
            print(f"Expert {exp_idx:<5} | {cnt:<14,} | {pct:<15.2f}% | {weight:<18.4f}")
        print(divider + "\n")


def load_validation_texts(
    data_path: str = "roneneldan/TinyStories",
    split: str = "validation",
    num_samples: int = 100,
    seed: int = 42,
) -> list[str]:
    """Loads validation texts from Hugging Face dataset or generates fallback synthetic samples."""
    texts = []
    try:
        from datasets import load_dataset

        logger.info("Loading dataset %r (split: %r)...", data_path, split)
        dataset = load_dataset(data_path)
        if split in dataset:
            raw_texts = dataset[split]["text"]
        elif "train" in dataset:
            raw_texts = dataset["train"]["text"]
        else:
            first_split = next(iter(dataset.keys()))
            raw_texts = dataset[first_split]["text"]

        random.seed(seed)
        if num_samples is not None and num_samples > 0 and len(raw_texts) > num_samples:
            texts = random.sample(raw_texts, num_samples)
        else:
            texts = list(raw_texts)
        logger.info("Successfully loaded %d texts from dataset %r", len(texts), data_path)
    except Exception as e:
        logger.warning(
            "Could not load Hugging Face dataset (%s). Generating %d synthetic text samples for analysis.",
            e,
            num_samples if (num_samples is not None and num_samples > 0) else 100,
        )
        random.seed(seed)
        num_synth = num_samples if (num_samples is not None and num_samples > 0) else 100
        sample_vocabulary = [
            "Once upon a time",
            "there was a little dragon who loved learning neural networks.",
            "Sparse mixture of experts routes tokens efficiently across experts.",
            "Mixtral MoE uses grouped query attention and top-2 softmax routing.",
            "The validation data is randomly sampled to inspect expert load balance.",
        ]
        texts = [
            " ".join(random.choices(sample_vocabulary, k=4)) for _ in range(num_synth)
        ]

    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Router Expert Token Routing Distribution in Sparse MoE Models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/mixtral_moe_tinystories",
        help="Path to checkpoint directory containing model.pth and config.pth",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mixtral_moe",
        help="Model architecture name in MODEL_REGISTRY",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Optional path to model config YAML file (used if no checkpoint is available)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="roneneldan/TinyStories",
        help="Path or name of dataset for validation sampling",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to sample from (default: validation)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of validation samples to evaluate (-1 or 0 to use all samples in dataset split)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data sampling",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional file path to save metrics JSON report",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running router analysis on device: %s", device)

    # 1. Load Model & Tokenizer
    tokenizer = None
    model = None

    if os.path.exists(args.model_path) and os.path.exists(
        os.path.join(args.model_path, "model.pth")
    ):
        logger.info("Loading trained model from checkpoint directory: %s", args.model_path)
        model = load_model(args.model_path, model_name=args.model_name, device=device)

        tok_path = os.path.join(args.model_path, "tokenizer.json")
        if os.path.exists(tok_path):
            logger.info("Loading tokenizer from: %s", tok_path)
            tokenizer = CharTokenizer.load(tok_path)
    elif args.model_config and os.path.exists(args.model_config):
        logger.info("Loading initialized model architecture from YAML config: %s", args.model_config)
        model = build_model(args.model_config).to(device)
    else:
        logger.warning(
            "Model checkpoint at %r not found. Falling back to default baseline config configs/mixtral_moe_config.yaml",
            args.model_path,
        )
        fallback_config = "configs/mixtral_moe_config.yaml"
        if os.path.exists(fallback_config):
            model = build_model(fallback_config).to(device)
        else:
            raise FileNotFoundError("No valid model checkpoint or config YAML file found.")

    # 2. Prepare Tokenizer if needed
    val_texts = load_validation_texts(
        data_path=args.data_path,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    if tokenizer is None:
        logger.info("Initializing and fitting CharTokenizer on validation samples...")
        max_vocab = getattr(model.config, "vocab_size", 256)
        tokenizer = CharTokenizer(max_vocab_size=max_vocab)
        tokenizer.fit(val_texts)

    # 3. Create Dataset & DataLoader
    block_size = getattr(model.config, "block_size", 1024)
    val_dataset = CausalLanguageModelingDataset(tokenizer, val_texts, block_size=block_size)
    val_dataloader = CausalLanguageModelingDataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 4. Attach Router Tracker Hooks
    tracker = RouterTracker(model)
    if not tracker.router_data:
        logger.warning("No TopKRouter layers found in the loaded model!")
        return

    logger.info(
        "Attached forward hooks to %d router layer(s): %s",
        len(tracker.router_data),
        list(tracker.router_data.keys()),
    )

    # 5. Run Forward Pass Inference
    model.eval()
    logger.info("Evaluating expert routing across %d validation batches...", len(val_dataloader))
    with torch.no_grad():
        for x, targets in val_dataloader:
            x = x.to(device)
            _ = model(x)

    # 6. Compute & Print Metrics
    metrics = tracker.compute_metrics()
    tracker.remove_hooks()

    print_analysis_report(metrics)

    if args.output_json:
        out_path = args.output_json
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Router analysis report saved to: %s", out_path)


if __name__ == "__main__":
    main()
