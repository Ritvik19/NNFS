import unittest
import torch
import torch.nn as nn

from nnfs.layers import RotaryEmbedding
from nnfs.models.llama3 import Llama3, Llama3Config
from nnfs.modules import Llama3TransformerBlock
from nnfs.utils.model_io import build_model
from nnfs.utils.text_generation import generate


class TestLlama3Components(unittest.TestCase):
    def test_llama3_config_defaults(self):
        config = Llama3Config()
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.block_size, 1024)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.n_layers, 4)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.n_kv_heads, 2)
        self.assertEqual(config.d_ff, 1024)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.rope_theta, 500000.0)

    def test_llama3_transformer_block(self):
        block = Llama3TransformerBlock(
            d_model=64,
            n_heads=8,
            n_kv_heads=2,
            d_ff=128,
            dropout=0.0,
            rope_theta=500000.0,
            rope_scaling={"rope_type": "llama3", "factor": 8.0},
        )
        x = torch.randn(2, 8, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 8, 64))
        self.assertIsNone(block.attn.q_proj.bias)
        self.assertIsNone(block.attn.k_proj.bias)
        self.assertIsNone(block.attn.v_proj.bias)
        self.assertIsNone(block.attn.out_proj.bias)
        self.assertIsNone(block.ffn.w_gate.bias)
        self.assertIsNone(block.ffn.w_up.bias)
        self.assertIsNone(block.ffn.w_down.bias)

    def test_llama3_rope_scaling(self):
        dim = 16
        base_rope = RotaryEmbedding(dim=dim, base=500000.0)
        scaled_rope = RotaryEmbedding(
            dim=dim,
            base=500000.0,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        )
        self.assertEqual(base_rope.inv_freq.shape, (dim // 2,))
        self.assertEqual(scaled_rope.inv_freq.shape, (dim // 2,))
        # Low frequency components (long wavelengths) should be scaled down by factor
        self.assertLessEqual(scaled_rope.inv_freq[-1].item(), base_rope.inv_freq[-1].item())


class TestLlama3Model(unittest.TestCase):
    def setUp(self):
        self.config = Llama3Config(
            vocab_size=100,
            block_size=32,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            dropout=0.0,
            rope_theta=500000.0,
            rope_scaling={"rope_type": "llama3", "factor": 8.0},
        )
        self.model = Llama3(self.config)

    def test_output_shape(self):
        batch_size, seq_len = 4, 16
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        logits = self.model(x)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))

    def test_weight_tying(self):
        torch.testing.assert_close(self.model.lm_head.weights, self.model.tok_embed.embed.t())
        self.assertIsNone(self.model.lm_head.bias)

    def test_gradient_flow(self):
        batch_size, seq_len = 2, 8
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        self.model.train()
        logits = self.model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.config.vocab_size), targets.view(-1)
        )
        loss.backward()

        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
            self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN")
            self.assertFalse(torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf")

    def test_causal_masking(self):
        self.model.eval()
        x1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        x2 = x1.clone()
        x2[0, 5] = 99

        with torch.no_grad():
            out1 = self.model(x1)
            out2 = self.model(x2)

        torch.testing.assert_close(out1[0, :5], out2[0, :5])
        self.assertFalse(torch.allclose(out1[0, 5:], out2[0, 5:]))

    def test_build_model_integration(self):
        model = build_model("configs/llama3_config.yaml")
        self.assertIsInstance(model, Llama3)
        x = torch.randint(0, 256, (2, 10))
        out = model(x)
        self.assertEqual(out.shape, (2, 10, 256))

    def test_autoregressive_generation(self):
        idx = torch.tensor([[1, 2, 3]])
        max_new_tokens = 5
        generated = generate(self.model, idx, max_new_tokens=max_new_tokens)
        self.assertEqual(generated.shape, (1, 3 + max_new_tokens))
        torch.testing.assert_close(generated[0, :3], idx[0])


if __name__ == "__main__":
    unittest.main()
