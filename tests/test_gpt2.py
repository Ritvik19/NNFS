import unittest

import torch
import torch.nn as nn

from nnfs.models.gpt1 import GPT1, GPT1Config
from nnfs.models.gpt2 import GPT2, GPT2Config
from nnfs.utils.model_io import build_model


class TestGPT2Model(unittest.TestCase):
    def setUp(self):
        self.config = GPT2Config(
            vocab_size=100,
            block_size=32,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
        )
        self.model = GPT2(self.config)

    def test_has_ln_f(self):
        self.assertTrue(hasattr(self.model, "ln_f"))
        self.assertEqual(self.model.ln_f.gamma.shape, (self.config.d_model,))

    def test_param_count_vs_gpt1(self):
        gpt1 = GPT1(
            GPT1Config(
                vocab_size=self.config.vocab_size,
                block_size=self.config.block_size,
                d_model=self.config.d_model,
                n_layers=self.config.n_layers,
                n_heads=self.config.n_heads,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout,
            )
        )
        gpt2_params = sum(p.numel() for p in self.model.parameters())
        gpt1_params = sum(p.numel() for p in gpt1.parameters())
        self.assertEqual(gpt2_params - gpt1_params, 2 * self.config.d_model)

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
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.config.vocab_size), targets.view(-1)
        )
        loss.backward()

        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
            self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN")
            self.assertFalse(torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf")

        self.assertIsNotNone(self.model.ln_f.gamma.grad)
        self.assertIsNotNone(self.model.ln_f.beta.grad)

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

    def test_weight_initialization(self):
        for name, param in self.model.named_parameters():
            if "bias" in name and param is not None:
                torch.testing.assert_close(
                    param, torch.zeros_like(param), msg=f"Bias {name} not zero-initialized"
                )
            elif "gamma" in name:
                torch.testing.assert_close(
                    param, torch.ones_like(param), msg=f"LayerNorm gamma {name} not 1.0 initialized"
                )
            elif "beta" in name:
                torch.testing.assert_close(
                    param, torch.zeros_like(param), msg=f"LayerNorm beta {name} not 0.0 initialized"
                )
            elif "weights" in name or "embed" in name:
                std = param.std().item()
                self.assertTrue(
                    0.01 < std < 0.03,
                    f"Weight {name} std {std} outside expected range around 0.02",
                )

    def test_build_model_integration(self):
        model = build_model("configs/gpt2_config.yaml")
        self.assertIsInstance(model, GPT2)
        self.assertTrue(hasattr(model, "ln_f"))
        x = torch.randint(0, 256, (2, 10))
        out = model(x)
        self.assertEqual(out.shape, (2, 10, 256))

    def test_weight_tying(self):
        torch.testing.assert_close(self.model.lm_head.weights, self.model.tok_embed.embed.t())
        self.assertIsNone(self.model.lm_head.bias)

    def test_autoregressive_generation(self):
        from nnfs.utils.text_generation import generate

        idx = torch.tensor([[1, 2, 3]])
        max_new_tokens = 5
        generated = generate(self.model, idx, max_new_tokens=max_new_tokens)
        self.assertEqual(generated.shape, (1, 3 + max_new_tokens))
        torch.testing.assert_close(generated[0, :3], idx[0])


if __name__ == "__main__":
    unittest.main()
