import argparse
import logging
import os
import sys
import yaml
import torch
import wandb

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from tqdm import trange
from nnfs.utils.model_io import build_model
from nnfs.preprocessors.char_tokenizer import CharTokenizer
from nnfs.utils.causal_language_modelling import (
    CausalLanguageModelingDataset,
    CausalLanguageModelingDataLoader,
    CausalLanguageModelingTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Train a Causal Language Model (GPT-1)")
    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to model config YAML file",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        help="Path to train config YAML file",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Override wandb run name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--lb-coef",
        type=float,
        default=None,
        help="Override auxiliary load balancing loss coefficient",
    )
    args = parser.parse_args()

    # Load configurations
    train_config = {}
    if args.train_config and os.path.exists(args.train_config):
        with open(args.train_config, "r", encoding="utf-8") as f:
            train_config = yaml.safe_load(f) or {}

    model_config = {}
    if args.model_config and os.path.exists(args.model_config):
        with open(args.model_config, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f) or {}

    batch_size = train_config.get("batch_size", 2)
    epochs = train_config.get("epochs", 5)
    learning_rate = train_config.get("learning_rate", 1e-4)
    max_vocab_size = train_config.get("max_vocab_size", 192)
    data_path = train_config.get("data_path")
    save_dir = train_config.get("save_dir", "checkpoints")
    load_balancing_coef = (
        args.lb_coef
        if args.lb_coef is not None
        else train_config.get("load_balancing_coef", 0.0)
    )

    use_wandb = train_config.get("wandb", True) and not args.no_wandb
    wandb_project = args.wandb_project or train_config.get("wandb_project", "nnfs")
    wandb_run_name = args.wandb_run_name or train_config.get("wandb_run_name")

    logger.info("Train config: %s", train_config)
    logger.info("Model config: %s", model_config)
    logger.info("Load balancing loss coefficient: %f", load_balancing_coef)

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={**train_config, **model_config, "load_balancing_coef": load_balancing_coef},
        )
        logger.info("wandb run initialized: %s / %s", wandb_project, wandb.run.name)

    # Load text dataset
    logger.info("Loading dataset: %s", data_path)
    dataset = load_dataset(data_path)
    train_texts = dataset["train"]["text"]
    val_texts = dataset["validation"]["text"]
    logger.info("Train samples: %d | Val samples: %d", len(train_texts), len(val_texts))

    # Initialize and fit tokenizer
    tokenizer = CharTokenizer(max_vocab_size=max_vocab_size)
    tokenizer.fit(train_texts)
    logger.info("Tokenizer vocab size: %d", tokenizer.vocab_size)

    # Build model using YAML config
    model = build_model(args.model_config)
    block_size = model.config.block_size
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %s | block_size: %d", f"{num_params:,}", block_size)

    if use_wandb:
        wandb.config.update({"num_params": num_params, "block_size": block_size}, allow_val_change=True)

    # Prepare dataset & dataloader
    train_dataset = CausalLanguageModelingDataset(tokenizer, train_texts, block_size=block_size)
    val_dataset = CausalLanguageModelingDataset(tokenizer, val_texts, block_size=block_size)
    train_dataloader = CausalLanguageModelingDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = CausalLanguageModelingDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup optimizer and trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = CausalLanguageModelingTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        load_balancing_coef=load_balancing_coef,
    )

    logger.info("Starting training for %d epochs on device: %s", epochs, trainer.device)
    global_step = 0
    log_every = train_config.get("log_every_steps", 1)

    def on_step(
        step: int, loss: float, ce_loss: float = 0.0, lb_loss: float = 0.0
    ) -> None:
        if not use_wandb or step % log_every != 0:
            return
        wandb.log(
            {
                "train/loss": loss,
                "train/ce_loss": ce_loss,
                "train/lb_loss": lb_loss,
                "lr": learning_rate,
            },
            step=step,
        )

    for epoch in trange(1, epochs + 1, desc="epochs"):
        train_loss, global_step = trainer.train_epoch(
            on_step=on_step,
            global_step=global_step,
        )
        val_loss = trainer.evaluate()
        logger.info(
            "Epoch %d/%d - train_loss: %.4f | val_loss: %.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
        )
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/loss": val_loss,
                },
                step=global_step,
            )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
        logger.info("Model and tokenizer saved to %s", save_dir)

    if use_wandb:
        wandb.finish()
        logger.info("wandb run finished")


if __name__ == "__main__":
    main()
