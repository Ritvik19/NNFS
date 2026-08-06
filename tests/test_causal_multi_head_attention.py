import unittest
import torch

from nnfs.layers import CausalMultiHeadAttention


class TestCausalMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.n_heads = 4
        self.attn = CausalMultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads, dropout=0.0)

    def test_output_shape(self):
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, self.d_model)
        out = self.attn(x)
        self.assertEqual(out.shape, (batch_size, seq_len, self.d_model))

    def test_causal_masking(self):
        """Verify that output at sequence index t is independent of future tokens at > t."""
        self.attn.eval()
        seq_len = 8
        x1 = torch.randn(1, seq_len, self.d_model)
        x2 = x1.clone()
        # Modify token at index 5
        x2[0, 5, :] += 5.0

        with torch.no_grad():
            out1 = self.attn(x1)
            out2 = self.attn(x2)

        # Timesteps 0..4 should be identical
        torch.testing.assert_close(out1[0, :5, :], out2[0, :5, :])
        # Timestep 5 onwards should differ
        self.assertFalse(torch.allclose(out1[0, 5:, :], out2[0, 5:, :]))

    def test_divisibility_assertion(self):
        with self.assertRaises(AssertionError):
            CausalMultiHeadAttention(d_model=63, n_heads=4)


if __name__ == "__main__":
    unittest.main()
