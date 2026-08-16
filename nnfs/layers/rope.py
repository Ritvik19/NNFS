import math
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key tensors.

    q shape: (B, n_heads, T, d_head)
    k shape: (B, 1, T, d_head) or (B, n_heads, T, d_head)
    cos, sin shape: (1, 1, T, d_head)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) generator with optional Llama 3 scaling."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
            factor = rope_scaling.get("factor", 8.0)
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            orig_max_pos = rope_scaling.get("original_max_position_embeddings", 8192)

            low_freq_w = orig_max_pos / low_freq_factor
            high_freq_w = orig_max_pos / high_freq_factor

            scaled_inv_freq = []
            for freq in inv_freq:
                w = 2.0 * math.pi / freq.item()
                if w < high_freq_w:
                    scaled_inv_freq.append(freq.item())
                elif w > low_freq_w:
                    scaled_inv_freq.append(freq.item() / factor)
                else:
                    smooth = (orig_max_pos / w - high_freq_factor) / (low_freq_factor - high_freq_factor)
                    scaled_freq = (1.0 - smooth) * (freq.item() / factor) + smooth * freq.item()
                    scaled_inv_freq.append(scaled_freq)

            inv_freq = torch.tensor(scaled_inv_freq, dtype=torch.float32)
        elif rope_scaling is not None and rope_scaling.get("rope_type") == "yarn":
            factor = rope_scaling.get("factor", 32.0)
            beta_fast = rope_scaling.get("beta_fast", 32.0)
            beta_slow = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos = rope_scaling.get("original_max_position_embeddings", 4096)

            low_freq_w = orig_max_pos / beta_slow
            high_freq_w = orig_max_pos / beta_fast

            scaled_inv_freq = []
            for freq in inv_freq:
                w = 2.0 * math.pi / freq.item()
                if w < high_freq_w:
                    scaled_inv_freq.append(freq.item())
                elif w > low_freq_w:
                    scaled_inv_freq.append(freq.item() / factor)
                else:
                    smooth = (orig_max_pos / w - beta_slow) / (beta_fast - beta_slow)
                    scaled_freq = (1.0 - smooth) * (freq.item() / factor) + smooth * freq.item()
                    scaled_inv_freq.append(scaled_freq)

            inv_freq = torch.tensor(scaled_inv_freq, dtype=torch.float32)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns cos and sin cached tensors sliced up to seq_len."""
        if seq_len > self.max_position_embeddings:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
        return self.cos_cached[:, :, :seq_len, :].to(device), self.sin_cached[:, :, :seq_len, :].to(device)

