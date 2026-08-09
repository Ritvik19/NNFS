import math

import torch
import torch.nn as nn


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Calculates head-specific slopes for ALiBi (Attention with Linear Biases).

    For power of 2 head count n:
        slopes = 2^(-8 * h / n) for h = 1..n
    For non-power of 2:
        takes slopes for closest smaller power of 2, plus extra slopes interpolated from 2*n_base.

    Args:
        n_heads: Number of attention heads.

    Returns:
        1D Tensor of shape (n_heads,) containing head slopes.
    """
    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        base_slopes = _get_slopes_power_of_2(closest_power_of_2)
        extra_slopes = _get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: n_heads - closest_power_of_2]
        slopes = base_slopes + extra_slopes

    return torch.tensor(slopes, dtype=torch.float32)


class ALiBiPositionalBias(nn.Module):
    """ALiBi (Attention with Linear Biases) positional bias generator.

    Ref: 'Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation'
         (Press et al., 2021) https://arxiv.org/abs/2108.12409

    Biases query-key attention dot products with a non-learned distance penalty:
        Score_{h, i, j} = (q_i . k_j) / sqrt(d_head) - m_h * (i - j)

    Args:
        n_heads: Number of attention heads.
        max_seq_len: Initial sequence length to precompute and cache.
    """

    def __init__(self, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        slopes = get_alibi_slopes(n_heads)
        self.register_buffer("slopes", slopes, persistent=False)

        bias = self._build_bias(max_seq_len, slopes)
        self.register_buffer("bias_cached", bias, persistent=False)

    def _build_bias(self, seq_len: int, slopes: torch.Tensor) -> torch.Tensor:
        """Constructs ALiBi bias matrix of shape (1, n_heads, seq_len, seq_len)."""
        pos_i = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        pos_j = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
        distance = pos_j - pos_i  # j - i <= 0 for causal positions i >= j

        bias = slopes.view(1, self.n_heads, 1, 1) * distance.view(1, 1, seq_len, seq_len)
        return bias

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Returns ALiBi bias tensor sliced up to seq_len.

        Args:
            seq_len: Sequence length for current attention computation.
            device: Target device.
            dtype: Target data type.

        Returns:
            Tensor of shape (1, n_heads, seq_len, seq_len) containing ALiBi biases.
        """
        if seq_len > self.max_seq_len:
            bias = self._build_bias(seq_len, self.slopes.to(device)).to(device=device, dtype=dtype)
            return bias
        return self.bias_cached[:, :, :seq_len, :seq_len].to(device=device, dtype=dtype)
