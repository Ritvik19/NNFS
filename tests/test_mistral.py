import unittest
import torch
import torch.nn as nn

from nnfs.models.mistral import Mistral, MistralConfig
from nnfs.modules import MistralTransformerBlock
from nnfs.utils.model_io import build_model
from nnfs.utils.text_generation import generate


class TestMistralComponents(unittest.TestCase):
    def test_mistral_config_defaults(self):
        config = MistralConfig()
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.block_size, 1024)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.n_layers, 4)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.n_kv_heads, 2)
        self.assertEqual(config.d_ff, 1024)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.rope_theta, 1000000.0)
        self.assertEqual(config.sliding_window, 4096)
        self.assertFalse(config.interleaved_sliding_window)

    def test_mistral_transformer_block(self):
        block = MistralTransformerBlock(
            d_model=64,
            n_heads=8,
            n_kv_heads=2,
            d_ff=128,
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
        self.assertIsNone(block.ffn.w_gate.bias)
        self.assertIsNone(block.ffn.w_up.bias)
        self.assertIsNone(block.ffn.w_down.bias)

    def test_mistral_sliding_window_masking(self):
        # Window size 2: Token 4 can attend to tokens 3 and 4, but NOT tokens 0, 1, 2.
        config = MistralConfig(
            vocab_size=100,
            block_size=16,
            d_model=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=64,
            dropout=0.0,
            rope_theta=10000.0,
            sliding_window=2,
        )
        model = Mistral(config)
        model.eval()

        x1 = torch.tensor([[10, 20, 30, 40, 50, 60]])
        x2 = x1.clone()
        x2[0, 0] = 99  # mutate token at position 0

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        # Token 0 change MUST NOT affect tokens at position >= 0 + window = 2
        # i.e., position 2 (window=[1,2]), position 3 (window=[2,3]), etc.
        torch.testing.assert_close(out1[0, 3:], out2[0, 3:])
        self.assertFalse(torch.allclose(out1[0, 0], out2[0, 0]))

    def test_mistral_interleaved_sliding_window(self):
        config = MistralConfig(
            vocab_size=100,
            block_size=16,
            d_model=32,
            n_layers=4,
            n_heads=2,
            n_kv_heads=1,
            d_ff=64,
            dropout=0.0,
            sliding_window=4,
            interleaved_sliding_window=True,
        )
        model = Mistral(config)
        self.assertEqual(model.blocks[0].attn.sliding_window, 4)
        self.assertIsNone(model.blocks[1].attn.sliding_window)
        self.assertEqual(model.blocks[2].attn.sliding_window, 4)
        self.assertIsNone(model.blocks[3].attn.sliding_window)


class TestMistralModel(unittest.TestCase):
    def setUp(self):
        self.config = MistralConfig(
            vocab_size=100,
            block_size=32,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            dropout=0.0,
            rope_theta=1000000.0,
            sliding_window=16,
        )
        self.model = Mistral(self.config)

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

    def test_build_model_integration(self):
        model = build_model("configs/mistral_config.yaml")
        self.assertIsInstance(model, Mistral)
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
