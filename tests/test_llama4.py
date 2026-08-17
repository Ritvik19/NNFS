import os
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn

from nnfs.layers import Llama4Attention, SharedSparseMoE
from nnfs.models.llama4 import Llama4, Llama4Config
from nnfs.modules import Llama4TransformerBlock
from nnfs.utils.model_io import build_model, load_model
from nnfs.utils.text_generation import generate


class TestLlama4Components(unittest.TestCase):
    def test_llama4_config_defaults_and_presets(self):
        config = Llama4Config()
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.block_size, 1024)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.n_layers, 4)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.n_kv_heads, 2)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.top_k_experts, 1)
        self.assertEqual(config.irope_ratio, 3)
        self.assertEqual(config.chunk_size, 256)
        self.assertEqual(config.rope_theta, 500000.0)
        self.assertEqual(config.temp_scaling, 1.0)

        # Scout preset verification
        scout_cfg = Llama4Config.llama_4_scout()
        self.assertEqual(scout_cfg.d_model, 5120)
        self.assertEqual(scout_cfg.n_layers, 48)
        self.assertEqual(scout_cfg.n_heads, 40)
        self.assertEqual(scout_cfg.n_kv_heads, 8)
        self.assertEqual(scout_cfg.num_experts, 16)
        self.assertEqual(scout_cfg.top_k_experts, 1)
        self.assertEqual(scout_cfg.block_size, 10485760)

        # Maverick preset verification
        mav_cfg = Llama4Config.llama_4_maverick()
        self.assertEqual(mav_cfg.d_model, 5120)
        self.assertEqual(mav_cfg.n_layers, 48)
        self.assertEqual(mav_cfg.n_heads, 40)
        self.assertEqual(mav_cfg.n_kv_heads, 8)
        self.assertEqual(mav_cfg.num_experts, 128)
        self.assertEqual(mav_cfg.top_k_experts, 1)
        self.assertEqual(mav_cfg.block_size, 1048576)

    def test_shared_sparse_moe(self):
        moe = SharedSparseMoE(
            d_model=64,
            d_ff=128,
            d_ff_shared=128,
            num_experts=4,
            top_k_experts=1,
            dropout=0.0,
        )
        x = torch.randn(2, 8, 64)
        out = moe(x)
        self.assertEqual(out.shape, (2, 8, 64))

        # Check router output caching
        self.assertIsNotNone(moe.router.last_router_logits)
        self.assertEqual(moe.router.last_router_logits.shape, (2, 8, 4))
        self.assertEqual(moe.router.last_top_k_indices.shape, (2, 8, 1))

    def test_llama4_attention_irope(self):
        # 1. RoPE Layer with chunked attention
        rope_attn = Llama4Attention(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            is_rope_layer=True,
            chunk_size=16,
            temp_scaling=1.0,
        )
        x = torch.randn(2, 32, 64)
        out_rope = rope_attn(x)
        self.assertEqual(out_rope.shape, (2, 32, 64))
        self.assertIsNotNone(rope_attn.rotary_emb)

        # 2. NoPE Layer with global attention
        nope_attn = Llama4Attention(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            is_rope_layer=False,
            chunk_size=16,
            temp_scaling=1.0,
        )
        out_nope = nope_attn(x)
        self.assertEqual(out_nope.shape, (2, 32, 64))
        self.assertIsNone(nope_attn.rotary_emb)

        # 3. Temperature scaling verification
        scaled_attn = Llama4Attention(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            temp_scaling=2.0,
        )
        self.assertAlmostEqual(
            scaled_attn.attn_scale,
            rope_attn.attn_scale / 2.0,
            places=5,
        )

    def test_llama4_transformer_block(self):
        block = Llama4TransformerBlock(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            num_experts=4,
            top_k_experts=1,
            dropout=0.0,
            is_rope_layer=True,
            chunk_size=32,
        )
        x = torch.randn(2, 16, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 16, 64))


class TestLlama4Model(unittest.TestCase):
    def setUp(self):
        self.config = Llama4Config(
            vocab_size=100,
            block_size=32,
            d_model=64,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            d_ff_shared=128,
            num_experts=4,
            top_k_experts=1,
            dropout=0.0,
            rope_theta=500000.0,
            irope_ratio=3,
            chunk_size=16,
            tie_word_embeddings=True,
        )
        self.model = Llama4(self.config)

    def test_output_shape(self):
        batch_size, seq_len = 2, 16
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        logits = self.model(x)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))

    def test_irope_layer_assignment(self):
        # In 4 layers with irope_ratio=3:
        # Layers 0, 1, 2 should be RoPE layers (True)
        # Layer 3 should be NoPE layer (False)
        self.assertTrue(self.model.blocks[0].is_rope_layer)
        self.assertTrue(self.model.blocks[1].is_rope_layer)
        self.assertTrue(self.model.blocks[2].is_rope_layer)
        self.assertFalse(self.model.blocks[3].is_rope_layer)

    def test_active_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        active_params = self.model.count_active_parameters()
        self.assertLess(active_params, total_params)
        self.assertEqual(active_params, 253760)

        # Baseline mini config check
        baseline_config = Llama4Config()
        baseline_model = Llama4(baseline_config)
        self.assertEqual(sum(p.numel() for p in baseline_model.parameters()), 29174016)
        self.assertEqual(baseline_model.count_active_parameters(), 7153920)

    def test_weight_tying(self):
        torch.testing.assert_close(self.model.lm_head.weights, self.model.tok_embed.embed.t())
        self.assertIsNone(self.model.lm_head.bias)

    def test_gradient_flow(self):
        batch_size, seq_len = 4, 16
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        self.model.train()
        logits, loss = self.model(x, targets=targets)
        self.assertIsNotNone(loss)
        loss.backward()

        # Core non-routed components must ALWAYS have gradients
        core_modules = [
            self.model.tok_embed,
            self.model.rms_f,
        ]
        for block in self.model.blocks:
            core_modules.extend([
                block.rms_1,
                block.attn,
                block.rms_2,
                block.moe.router,
                block.moe.shared_expert,
            ])

        for mod in core_modules:
            for name, param in mod.named_parameters():
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
                self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN")
                self.assertFalse(torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf")

        # For routed experts, verify that any assigned expert has valid gradients
        for block in self.model.blocks:
            for exp in block.moe.routed_experts:
                for name, param in exp.named_parameters():
                    if param.grad is not None:
                        self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN")
                        self.assertFalse(torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf")

    def test_causal_masking(self):
        self.model.eval()
        x1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        x2 = x1.clone()
        x2[0, 4] = 99

        with torch.no_grad():
            out1 = self.model(x1)
            out2 = self.model(x2)

        # Tokens before index 4 should have identical logits
        torch.testing.assert_close(out1[0, :4], out2[0, :4])
        self.assertFalse(torch.allclose(out1[0, 4:], out2[0, 4:]))

    def test_router_outputs(self):
        x = torch.randint(0, self.config.vocab_size, (2, 8))
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        router_outs = self.model.get_router_outputs()
        self.assertEqual(len(router_outs), self.config.n_layers)
        for logits, indices in router_outs:
            self.assertEqual(logits.shape, (2, 8, self.config.num_experts))
            self.assertEqual(indices.shape, (2, 8, self.config.top_k_experts))

    def test_build_model_integration(self):
        model = build_model("configs/llama4_moe_config.yaml")
        self.assertIsInstance(model, Llama4)
        x = torch.randint(0, 256, (2, 10))
        out = model(x)
        self.assertEqual(out.shape, (2, 10, 256))

    def test_autoregressive_generation(self):
        idx = torch.tensor([[1, 2, 3]])
        max_new_tokens = 5
        generated = generate(self.model, idx, max_new_tokens=max_new_tokens)
        self.assertEqual(generated.shape, (1, 3 + max_new_tokens))
        torch.testing.assert_close(generated[0, :3], idx[0])

    def test_save_load_pretrained(self):
        temp_dir = tempfile.mkdtemp()
        try:
            self.model.save_pretrained(temp_dir)
            loaded_model = load_model(temp_dir, model_name="llama4_moe", device="cpu")
            self.assertIsInstance(loaded_model, Llama4)

            x = torch.randint(0, self.config.vocab_size, (2, 8))
            self.model.eval()
            loaded_model.eval()
            with torch.no_grad():
                out1 = self.model(x)
                out2 = loaded_model(x)
            torch.testing.assert_close(out1, out2)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
