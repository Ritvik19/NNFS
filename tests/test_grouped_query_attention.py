import unittest
import torch

from nnfs.layers import GroupedQueryAttention


class TestGroupedQueryAttention(unittest.TestCase):
    def test_gqa_output_shape(self):
        B, T, d_model = 2, 8, 64
        n_heads = 8
        n_kv_heads = 2
        gqa = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, use_rope=True)

        x = torch.randn(B, T, d_model)
        out = gqa(x)
        self.assertEqual(out.shape, (B, T, d_model))

    def test_mha_fallback(self):
        B, T, d_model = 2, 8, 64
        n_heads = 4
        gqa = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=4, use_rope=True)

        x = torch.randn(B, T, d_model)
        out = gqa(x)
        self.assertEqual(out.shape, (B, T, d_model))
        self.assertEqual(gqa.n_rep, 1)

    def test_mqa_fallback(self):
        B, T, d_model = 2, 8, 64
        n_heads = 4
        gqa = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=1, use_rope=True)

        x = torch.randn(B, T, d_model)
        out = gqa(x)
        self.assertEqual(out.shape, (B, T, d_model))
        self.assertEqual(gqa.n_rep, 4)

    def test_gradient_flow(self):
        B, T, d_model = 2, 8, 64
        gqa = GroupedQueryAttention(d_model=d_model, n_heads=8, n_kv_heads=2)

        x = torch.randn(B, T, d_model, requires_grad=True)
        out = gqa(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for name, param in gqa.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
            self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} has NaN")


if __name__ == "__main__":
    unittest.main()
