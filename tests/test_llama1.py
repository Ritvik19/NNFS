import unittest
import torch
import torch.nn as nn

from nnfs.models.llama1 import Llama1, Llama1Config
from nnfs.modules import Llama1TransformerBlock
from nnfs.utils.model_io import build_model
from nnfs.utils.text_generation import generate


class TestLlama1Components(unittest.TestCase):
    def test_llama1_config_defaults(self):
        config = Llama1Config(d_model=512)
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.block_size, 512)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.n_layers, 6)
        self.assertEqual(config.n_heads, 8)
        # 2/3 * 4 * 512 = 1365.33 -> rounded up to multiple of 256 is 1536
        self.assertEqual(config.d_ff, 1536)
        self.assertEqual(config.dropout, 0.0)

    def test_llama1_transformer_block(self):
        block = Llama1TransformerBlock(d_model=64, n_heads=4, d_ff=128, dropout=0.0)
        x = torch.randn(2, 8, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 8, 64))
        self.assertIsNone(block.attn.qkv.bias)
        self.assertIsNone(block.attn.out.bias)
        self.assertIsNone(block.ffn.w_gate.bias)
        self.assertIsNone(block.ffn.w_up.bias)
        self.assertIsNone(block.ffn.w_down.bias)


class TestLlama1Model(unittest.TestCase):
    def setUp(self):
        self.config = Llama1Config(
            vocab_size=100,
            block_size=32,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.0,
        )
        self.model = Llama1(self.config)

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
        model = build_model("configs/llama1_config.yaml")
        self.assertIsInstance(model, Llama1)
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
