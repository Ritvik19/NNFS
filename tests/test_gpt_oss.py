import unittest
import torch
import torch.nn as nn

from nnfs.layers.gpt_oss_attention import GptOssAttention
from nnfs.models.gpt_oss import GptOss, GptOssConfig
from nnfs.modules.gpt_oss_transformer_block import GptOssTransformerBlock
from nnfs.utils.model_io import build_model
from nnfs.utils.text_generation import generate


class TestGptOssComponents(unittest.TestCase):
    def test_gpt_oss_config_defaults(self):
        config = GptOssConfig()
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.block_size, 1024)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.n_layers, 4)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.n_kv_heads, 2)
        self.assertEqual(config.d_head, 64)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.top_k_experts, 2)
        self.assertEqual(config.d_ff, 256)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.rope_theta, 150000.0)
        self.assertEqual(config.sliding_window, 256)
        self.assertTrue(config.interleaved_sliding_window)
        self.assertEqual(config.swiglu_limit, 7.0)
        self.assertTrue(config.attention_bias)
        self.assertTrue(config.sink_bias)
        self.assertFalse(config.tie_word_embeddings)

    def test_gpt_oss_presets(self):
        config_20b = GptOssConfig.gpt_oss_20b()
        self.assertEqual(config_20b.n_layers, 24)
        self.assertEqual(config_20b.num_experts, 32)
        self.assertEqual(config_20b.top_k_experts, 4)
        self.assertEqual(config_20b.d_model, 2880)
        self.assertEqual(config_20b.n_heads, 64)
        self.assertEqual(config_20b.n_kv_heads, 8)

        config_120b = GptOssConfig.gpt_oss_120b()
        self.assertEqual(config_120b.n_layers, 36)
        self.assertEqual(config_120b.num_experts, 128)
        self.assertEqual(config_120b.top_k_experts, 4)
        self.assertEqual(config_120b.d_model, 2880)
        self.assertEqual(config_120b.n_heads, 64)
        self.assertEqual(config_120b.n_kv_heads, 8)

    def test_attention_sink_mechanism(self):
        attn = GptOssAttention(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_head=16,
            dropout=0.0,
            use_rope=False,
            sink_bias=True,
        )
        self.assertIsNotNone(attn.sink_bias)
        self.assertEqual(attn.sink_bias.shape, (4,))

        x = torch.randn(2, 8, 64, requires_grad=True)
        out = attn(x)
        self.assertEqual(out.shape, (2, 8, 64))

        # Check gradient flow to sink_bias
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(attn.sink_bias.grad)
        self.assertIsNotNone(x.grad)

    def test_attention_sink_null_behavior(self):
        attn = GptOssAttention(
            d_model=32,
            n_heads=2,
            n_kv_heads=2,
            d_head=16,
            dropout=0.0,
            use_rope=False,
            bias=False,
            sink_bias=True,
        )
        x = torch.randn(1, 4, 32)
        # Force sink_bias to very large positive value -> attention weights should collapse to 0
        with torch.no_grad():
            attn.sink_bias.fill_(1000.0)
        out_sink = attn(x)

        torch.testing.assert_close(out_sink, torch.zeros_like(out_sink), atol=1e-5, rtol=1e-5)

    def test_gpt_oss_transformer_block(self):
        block = GptOssTransformerBlock(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_head=16,
            d_ff=64,
            num_experts=4,
            top_k_experts=2,
            dropout=0.0,
            rope_theta=150000.0,
            sliding_window=4,
            swiglu_limit=7.0,
            attention_bias=True,
            sink_bias=True,
        )
        x = torch.randn(2, 8, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 8, 64))
        self.assertIsNotNone(block.attn.q_proj.bias)
        self.assertIsNotNone(block.attn.k_proj.bias)
        self.assertIsNotNone(block.attn.v_proj.bias)
        self.assertIsNotNone(block.attn.out_proj.bias)


class TestGptOssModel(unittest.TestCase):
    def setUp(self):
        self.config = GptOssConfig(
            vocab_size=100,
            block_size=16,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_head=8,
            num_experts=4,
            top_k_experts=2,
            d_ff=32,
            dropout=0.0,
            rope_theta=150000.0,
            sliding_window=4,
            interleaved_sliding_window=True,
            swiglu_limit=7.0,
            attention_bias=True,
            sink_bias=True,
            tie_word_embeddings=False,
        )

    def test_active_parameter_count(self):
        model = GptOss(self.config)
        total_params = sum(p.numel() for p in model.parameters())
        active_params = model.count_active_parameters()
        self.assertLess(active_params, total_params)

        # Baseline mini config check
        baseline_config = GptOssConfig()
        baseline_model = GptOss(baseline_config)
        total_baseline = sum(p.numel() for p in baseline_model.parameters())
        active_baseline = baseline_model.count_active_parameters()
        self.assertGreater(total_baseline, active_baseline)

    def test_forward_pass_logits(self):
        model = GptOss(self.config)
        model.eval()
        idx = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            logits = model(idx)
        self.assertEqual(logits.shape, (2, 8, 100))

    def test_forward_pass_with_targets(self):
        model = GptOss(self.config)
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

    def test_build_model_gpt_oss(self):
        model = build_model("configs/gpt_oss_moe_config.yaml")
        self.assertIsInstance(model, GptOss)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)
        self.assertGreater(model.count_active_parameters(), 0)

    def test_text_generation(self):
        model = GptOss(self.config)
        model.eval()
        idx = torch.tensor([[1, 2, 3]])
        max_new_tokens = 5
        generated = generate(model, idx, max_new_tokens=max_new_tokens)
        self.assertEqual(generated.shape, (1, 3 + max_new_tokens))
        torch.testing.assert_close(generated[0, :3], idx[0])


if __name__ == "__main__":
    unittest.main()
