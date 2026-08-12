import os
import tempfile
import unittest

import torch
import torch.nn as nn

from nnfs.activations import GELU, ReLU, SwiGLU
from nnfs.layers import SinusoidalPositionalEncoding
from nnfs.models import Transformer, TransformerConfig
from nnfs.modules import TransformerBlock
from nnfs.utils import build_model, generate


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.config = TransformerConfig(
            vocab_size=100,
            block_size=32,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            norm_first=False,
            positional_encoding="sinusoidal",
            activation="relu",
        )
        self.model = Transformer(self.config)

    def test_sinusoidal_positional_encoding(self):
        pe_layer = SinusoidalPositionalEncoding(max_len=32, d_model=64)
        pe_out = pe_layer(16)
        self.assertEqual(pe_out.shape, (16, 64))
        # Verify sine/cosine alternating indices
        self.assertAlmostEqual(pe_out[0, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(pe_out[0, 1].item(), 1.0, places=5)

    def test_transformer_block(self):
        block_post_ln = TransformerBlock(d_model=64, n_heads=4, d_ff=128, norm_first=False)
        block_pre_ln = TransformerBlock(d_model=64, n_heads=4, d_ff=128, norm_first=True)

        x = torch.randn(2, 8, 64)
        out_post = block_post_ln(x)
        out_pre = block_pre_ln(x)

        self.assertEqual(out_post.shape, (2, 8, 64))
        self.assertEqual(out_pre.shape, (2, 8, 64))

    def test_output_shape(self):
        batch_size, seq_len = 4, 16
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        logits = self.model(x)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))

    def test_gradient_flow(self):
        batch_size, seq_len = 2, 8
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        self.model.train()
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        loss.backward()

        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
            self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN")
            self.assertFalse(torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf")

    def test_causal_masking(self):
        """Verify that prediction for token at index t does not depend on tokens at indices > t."""
        self.model.eval()
        x1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        x2 = x1.clone()
        x2[0, 5] = 99  # Modify index 5

        with torch.no_grad():
            out1 = self.model(x1)
            out2 = self.model(x2)

        # Output up to index 4 must be identical
        torch.testing.assert_close(out1[0, :5], out2[0, :5])
        # Output at index 5 and beyond should differ
        self.assertFalse(torch.allclose(out1[0, 5:], out2[0, 5:]))

    def test_weight_tying(self):
        """Verify weight tying between tok_embed and lm_head."""
        torch.testing.assert_close(self.model.lm_head.weights, self.model.tok_embed.embed.t())
        self.assertIsNone(self.model.lm_head.bias)

    def test_save_and_load(self):
        """Verify model save_pretrained and load_pretrained functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.model.save_pretrained(tmp_dir)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "config.pth")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model.pth")))

            loaded_model = Transformer(self.config)
            loaded_model.load_pretrained(tmp_dir)

            x = torch.randint(0, self.config.vocab_size, (2, 8))
            self.model.eval()
            loaded_model.eval()
            with torch.no_grad():
                out_orig = self.model(x)
                out_loaded = loaded_model(x)
            torch.testing.assert_close(out_orig, out_loaded)

    def test_build_model_integration(self):
        config_files = [
            "configs/transformer_config.yaml",
            "configs/transformer_mqa_config.yaml",
            "configs/transformer_gqa_config.yaml",
        ]
        for cfg_file in config_files:
            with self.subTest(config_file=cfg_file):
                model = build_model(cfg_file)
                self.assertIsInstance(model, Transformer)

                x = torch.randint(0, 256, (2, 10))
                out = model(x)
                self.assertEqual(out.shape, (2, 10, 256))

    def test_autoregressive_generation(self):
        idx = torch.tensor([[1, 2, 3]])
        max_new_tokens = 5
        generated = generate(self.model, idx, max_new_tokens=max_new_tokens)
        self.assertEqual(generated.shape, (1, 3 + max_new_tokens))
        torch.testing.assert_close(generated[0, :3], idx[0])

    def test_configurable_positional_encodings(self):
        encodings = ["sinusoidal", "learned", "alibi", "rope", "none"]
        for enc in encodings:
            with self.subTest(positional_encoding=enc):
                config = TransformerConfig(
                    vocab_size=100,
                    block_size=32,
                    d_model=64,
                    n_layers=2,
                    n_heads=4,
                    d_ff=128,
                    positional_encoding=enc,
                )
                model = Transformer(config)
                x = torch.randint(0, 100, (2, 8))
                out = model(x)
                self.assertEqual(out.shape, (2, 8, 100))

    def test_configurable_activation_functions(self):
        activations = ["relu", "gelu", "swiglu"]
        for act in activations:
            with self.subTest(activation=act):
                config = TransformerConfig(
                    vocab_size=100,
                    block_size=32,
                    d_model=64,
                    n_layers=2,
                    n_heads=4,
                    d_ff=128,
                    activation=act,
                )
                model = Transformer(config)
                x = torch.randint(0, 100, (2, 8))
                out = model(x)
                self.assertEqual(out.shape, (2, 8, 100))

    def test_transformer_block_activations(self):
        x = torch.randn(2, 8, 64)
        for act in ["relu", "gelu", "swiglu", GELU()]:
            block = TransformerBlock(d_model=64, n_heads=4, d_ff=128, activation=act)
            out = block(x)
            self.assertEqual(out.shape, (2, 8, 64))

    def test_configurable_attention_types(self):
        attn_configs = [
            ("mha", None),
            ("mqa", 1),
            ("gqa", 2),
        ]
        for attn_type, n_kv_heads in attn_configs:
            with self.subTest(attn_type=attn_type):
                config = TransformerConfig(
                    vocab_size=100,
                    block_size=32,
                    d_model=64,
                    n_layers=2,
                    n_heads=4,
                    n_kv_heads=n_kv_heads,
                    attn_type=attn_type,
                    d_ff=128,
                )
                model = Transformer(config)
                x = torch.randint(0, 100, (2, 8))
                out = model(x)
                self.assertEqual(out.shape, (2, 8, 100))


if __name__ == "__main__":
    unittest.main()
