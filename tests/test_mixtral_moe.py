import unittest
import torch
import torch.nn as nn

from nnfs.layers.sparse_moe import SparseMoE, TopKRouter
from nnfs.models.mixtral_moe import MixtralMoE, MixtralMoEConfig
from nnfs.modules import MixtralTransformerBlock
from nnfs.utils.model_io import build_model
from nnfs.utils.text_generation import generate


class TestMixtralMoEComponents(unittest.TestCase):
    def test_mixtral_moe_config_defaults(self):
        config = MixtralMoEConfig()
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.block_size, 1024)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.n_layers, 4)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.n_kv_heads, 2)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.top_k_experts, 2)
        self.assertEqual(config.d_ff, 1024)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.rope_theta, 1000000.0)
        self.assertEqual(config.sliding_window, 4096)
        self.assertFalse(config.interleaved_sliding_window)

    def test_top_k_router(self):
        router = TopKRouter(d_model=64, num_experts=8, top_k=2)
        x = torch.randn(2, 5, 64)
        weights, indices = router(x)
        self.assertEqual(weights.shape, (2, 5, 2))
        self.assertEqual(indices.shape, (2, 5, 2))
        # Routing weights for top-k should sum to 1.0 per token
        sum_weights = weights.sum(dim=-1)
        torch.testing.assert_close(sum_weights, torch.ones_like(sum_weights))

    def test_sparse_moe_layer(self):
        moe = SparseMoE(d_model=64, d_ff=128, num_experts=4, top_k_experts=2, dropout=0.0)
        x = torch.randn(2, 6, 64, requires_grad=True)
        out = moe(x)
        self.assertEqual(out.shape, (2, 6, 64))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_mixtral_transformer_block(self):
        block = MixtralTransformerBlock(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            num_experts=4,
            top_k_experts=2,
            dropout=0.0,
            rope_theta=1000000.0,
            sliding_window=4,
        )
        x = torch.randn(2, 8, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 8, 64))
        self.assertIsNone(block.attn.q_proj.bias)
        self.assertIsNone(block.attn.k_proj.bias)
        self.assertIsNone(block.attn.v_proj.bias)
        self.assertIsNone(block.attn.out_proj.bias)


class TestMixtralMoEModel(unittest.TestCase):
    def setUp(self):
        self.config = MixtralMoEConfig(
            vocab_size=100,
            block_size=16,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            num_experts=4,
            top_k_experts=2,
            d_ff=64,
            dropout=0.0,
            rope_theta=10000.0,
            sliding_window=8,
        )

    def test_active_parameter_count(self):
        model = MixtralMoE(self.config)
        total_params = sum(p.numel() for p in model.parameters())
        active_params = model.count_active_parameters()
        self.assertLess(active_params, total_params)

        # Baseline mini config check
        baseline_config = MixtralMoEConfig()
        baseline_model = MixtralMoE(baseline_config)
        self.assertEqual(sum(p.numel() for p in baseline_model.parameters()), 26028288)
        self.assertEqual(baseline_model.count_active_parameters(), 7153920)

    def test_forward_pass_logits(self):
        model = MixtralMoE(self.config)
        model.eval()
        idx = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            logits = model(idx)
        self.assertEqual(logits.shape, (2, 8, 100))

    def test_forward_pass_with_targets(self):
        model = MixtralMoE(self.config)
        idx = torch.randint(0, 100, (2, 8))
        targets = torch.randint(0, 100, (2, 8))
        logits, loss = model(idx, targets=targets)
        self.assertEqual(logits.shape, (2, 8, 100))
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

        # Test autograd
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_build_model_mixtral_moe(self):
        model = build_model("configs/mixtral_moe_config.yaml")
        self.assertIsInstance(model, MixtralMoE)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)
        self.assertGreater(model.count_active_parameters(), 0)

    def test_text_generation(self):
        model = MixtralMoE(self.config)
        model.eval()
        idx = torch.tensor([[1, 2, 3]])
        max_new_tokens = 5
        generated = generate(model, idx, max_new_tokens=max_new_tokens)
        self.assertEqual(generated.shape, (1, 3 + max_new_tokens))
        torch.testing.assert_close(generated[0, :3], idx[0])


if __name__ == "__main__":
    unittest.main()
