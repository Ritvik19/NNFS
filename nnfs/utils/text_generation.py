import torch
import torch.nn as nn
from typing import Union
from nnfs.preprocessors.char_tokenizer import CharTokenizer
from nnfs.models.gpt1 import GPT1, GPT1Config
from nnfs.models.gpt2 import GPT2, GPT2Config

Tokenizer = Union[CharTokenizer]
Model = Union[GPT1, GPT2]
Config = Union[GPT1Config, GPT2Config]


def generate(
    model: Model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    model.eval()
    block_size = model.config.block_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = torch.softmax(logits, dim=-1)

                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    probs_sum = probs.sum(dim=-1, keepdim=True)
                    probs = torch.where(
                        probs_sum > 0, probs / probs_sum, torch.full_like(probs, 1.0 / probs.size(-1))
                    )

                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text(
    model: Model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    model.eval()
    device = next(model.parameters()).device

    encoded_prompt = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    idx = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

    generated_idx = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return tokenizer.decode(generated_idx[0].tolist(), skip_special_tokens=True)