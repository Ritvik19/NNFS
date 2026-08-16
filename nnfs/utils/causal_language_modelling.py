import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional, Callable
from nnfs.preprocessors.char_tokenizer import CharTokenizer
from nnfs.models.gpt1 import GPT1, GPT1Config
from nnfs.models.gpt2 import GPT2, GPT2Config
from nnfs.losses import CrossEntropy, LoadBalancingLoss
from tqdm import tqdm

Tokenizer = Union[CharTokenizer]
Model = Union[GPT1, GPT2]
Config = Union[GPT1Config, GPT2Config]

class CausalLanguageModelingDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, texts: list[str], block_size: int):
        self.tokenizer = tokenizer
        self.texts = texts
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        encoded = self.tokenizer.encode(text, add_bos=True, add_eos=True, sequence_length=self.block_size + 1)
        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        return x, y

    @staticmethod
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        return x, y

class CausalLanguageModelingDataLoader(DataLoader):
    def __init__(
        self,
        dataset: CausalLanguageModelingDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=CausalLanguageModelingDataset.collate_fn if hasattr(dataset, "collate_fn") else None,
            **kwargs,
        )

class CausalLanguageModelingTrainer:
    def __init__(
        self,
        model: Model,
        optimizer: Optional[Union[torch.optim.Optimizer, Tokenizer]] = None,
        train_dataloader: Optional[Union[DataLoader, CausalLanguageModelingDataset]] = None,
        val_dataloader: Optional[DataLoader] = None,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 1,
        shuffle: bool = True,
        lr: float = 1e-3,
        load_balancing_coef: float = 0.0,
    ):
        self.model = model
        self.load_balancing_coef = float(load_balancing_coef)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
            dataset_or_dl = train_dataloader
        else:
            # Flexible initialization support for legacy signature (model, tokenizer, dataset, batch_size)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            dataset_or_dl = train_dataloader if train_dataloader is not None else optimizer

        if isinstance(dataset_or_dl, DataLoader):
            self.train_dataloader = dataset_or_dl
        elif isinstance(dataset_or_dl, CausalLanguageModelingDataset):
            self.train_dataloader = CausalLanguageModelingDataLoader(
                dataset_or_dl, batch_size=batch_size, shuffle=shuffle
            )
        elif dataset_or_dl is not None:
            self.train_dataloader = DataLoader(dataset_or_dl, batch_size=batch_size, shuffle=shuffle)
        else:
            self.train_dataloader = None

        self.val_dataloader = val_dataloader
        self.loss_fn = CrossEntropy()
        self.lb_loss_fn = LoadBalancingLoss()

    def train_epoch(
        self,
        on_step: Optional[Callable] = None,
        global_step: int = 0,
    ) -> tuple[float, int]:
        if self.train_dataloader is None:
            raise ValueError("train_dataloader is not set.")
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(self.train_dataloader, desc="train", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            ce_loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            
            lb_loss_val = 0.0
            if self.load_balancing_coef > 0:
                router_outputs = None
                if hasattr(self.model, "get_router_outputs"):
                    router_outputs = self.model.get_router_outputs()
                if router_outputs:
                    lb_loss = self.lb_loss_fn(router_outputs)
                    lb_loss_val = lb_loss.item()
                    loss = ce_loss + self.load_balancing_coef * lb_loss
                else:
                    loss = ce_loss
            else:
                loss = ce_loss

            loss.backward()
            self.optimizer.step()
            step_loss = loss.item()
            total_loss += step_loss
            num_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{step_loss:.4f}", step=global_step)
            if on_step is not None:
                try:
                    on_step(global_step, step_loss, ce_loss.item(), lb_loss_val)
                except TypeError:
                    on_step(global_step, step_loss)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, global_step

    def train(self, epochs: int = 1) -> list[float]:
        epoch_losses = []
        global_step = 0
        for _ in range(epochs):
            loss, global_step = self.train_epoch(global_step=global_step)
            epoch_losses.append(loss)
        return epoch_losses

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> float:
        self.model.eval()
        target_dl = dataloader if dataloader is not None else self.val_dataloader
        if target_dl is None:
            target_dl = self.train_dataloader
        if target_dl is None:
            raise ValueError("No dataloader provided for evaluation.")

        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for x, y in target_dl:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0.0