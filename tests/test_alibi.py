import math
import unittest

import torch

from nnfs.layers import ALiBiPositionalBias, CausalMultiHeadAttention, get_alibi_slopes


class TestALiBi(unittest.TestCase):
    def test_get_alibi_slopes_power_of_two(self):
        slopes_8 = get_alibi_slopes(8)
        expected_8 = torch.tensor([2 ** (-i) for i in range(1, 9)], dtype=torch.float32)
        self.assertEqual(slopes_8.shape, (8,))
        torch.testing.assert_close(slopes_8, expected_8)

        slopes_16 = get_alibi_slopes(16)
        expected_16 = torch.tensor([2 ** (-0.5 * i) for i in range(1, 17)], dtype=torch.float32)
        self.assertEqual(slopes_16.shape, (16,))
        torch.testing.assert_close(slopes_16, expected_16)

    def test_get_alibi_slopes_non_power_of_two(self):
        slopes_12 = get_alibi_slopes(12)
        self.assertEqual(slopes_12.shape, (12,))
        self.assertTrue(torch.all(slopes_12 > 0))
        self.assertTrue(torch.all(slopes_12 < 1.0))

    def test_alibi_bias_shape_and_values(self):
        n_heads = 4
        seq_len = 8
        alibi = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=16)
        device = torch.device("cpu")
        bias = alibi(seq_len, device)

        self.assertEqual(bias.shape, (1, n_heads, seq_len, seq_len))

        # Diagonal (i == j) distance is 0 -> bias must be 0
        for h in range(n_heads):
            diag = torch.diagonal(bias[0, h], dim1=0, dim2=1)
            torch.testing.assert_close(diag, torch.zeros(seq_len))

        # Off-diagonal (i > j, key precedes query by 1 position: i - j = 1) -> bias must be -m_h
        slopes = get_alibi_slopes(n_heads)
        for h in range(n_heads):
            m_h = slopes[h].item()
            subdiag = torch.diagonal(bias[0, h], offset=-1)
            expected_subdiag = torch.full_like(subdiag, -m_h)
            torch.testing.assert_close(subdiag, expected_subdiag)

    def test_alibi_dynamic_extrapolation(self):
        alibi = ALiBiPositionalBias(n_heads=4, max_seq_len=16)
        device = torch.device("cpu")

        # Sequence length greater than cached max_seq_len
        extrapolated_seq_len = 32
        bias = alibi(extrapolated_seq_len, device)

        self.assertEqual(bias.shape, (1, 4, 32, 32))
        m_0 = alibi.slopes[0].item()
        self.assertTrue(math.isclose(bias[0, 0, 5, 0].item(), -5.0 * m_0, rel_tol=1e-5))

    def test_alibi_zero_parameters(self):
        alibi = ALiBiPositionalBias(n_heads=8, max_seq_len=1024)
        param_count = sum(p.numel() for p in alibi.parameters())
        self.assertEqual(param_count, 0)

    def test_alibi_causal_mha_integration(self):
        B, T, C = 2, 16, 64
        n_heads = 4
        x = torch.randn(B, T, C, requires_grad=True)

        attn = CausalMultiHeadAttention(d_model=C, n_heads=n_heads)
        alibi = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=64)

        alibi_bias = alibi(T, x.device)
        out = attn(x, alibi_bias=alibi_bias)

        self.assertEqual(out.shape, (B, T, C))

        # Test backward pass gradient propagation
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (B, T, C))


if __name__ == "__main__":
    unittest.main()
