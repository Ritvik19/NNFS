import unittest
import torch
import torch.nn as nn

from nnfs.models.gpt1 import GPT1, GPT1Config
from nnfs.utils.model_io import build_model


class TestGPT1Model(unittest.TestCase):
    def setUp(self):
        self.config = GPT1Config(
            vocab_size=100,
            block_size=32,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
        )
        self.model = GPT1(self.config)

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
        seq_len = 10
        x1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        x2 = x1.clone()
        # Change token at index 5
        x2[0, 5] = 99

        with torch.no_grad():
            out1 = self.model(x1)
            out2 = self.model(x2)

        # Output up to index 4 should be identical
        torch.testing.assert_close(out1[0, :5], out2[0, :5])
        # Output at index 5 and beyond can differ
        self.assertFalse(torch.allclose(out1[0, 5:], out2[0, 5:]))

    def test_weight_initialization(self):
        """Check that biases are 0.0 and weights are initialized with std around 0.02."""
        for name, param in self.model.named_parameters():
            if "bias" in name and param is not None:
                torch.testing.assert_close(param, torch.zeros_like(param), msg=f"Bias {name} not zero-initialized")
            elif "gamma" in name:
                torch.testing.assert_close(param, torch.ones_like(param), msg=f"LayerNorm gamma {name} not 1.0 initialized")
            elif "beta" in name:
                torch.testing.assert_close(param, torch.zeros_like(param), msg=f"LayerNorm beta {name} not 0.0 initialized")
            elif "weights" in name or "embed" in name:
                std = param.std().item()
                self.assertTrue(0.01 < std < 0.03, f"Weight {name} std {std} outside expected range around 0.02")

    def test_build_model_integration(self):
        model = build_model("configs/gpt1_config.yaml")
        self.assertIsInstance(model, GPT1)
        x = torch.randint(0, 256, (2, 10))
        out = model(x)
        self.assertEqual(out.shape, (2, 10, 256))

    def test_weight_tying(self):
        """Verify weight tying between tok_embed and lm_head."""
        torch.testing.assert_close(self.model.lm_head.weights, self.model.tok_embed.embed.t())
        self.assertIsNone(self.model.lm_head.bias)

    def test_autoregressive_generation(self):
        """Verify autoregressive token generation using generate utility function."""
        from nnfs.utils.text_generation import generate
        idx = torch.tensor([[1, 2, 3]])
        max_new_tokens = 5
        generated = generate(self.model, idx, max_new_tokens=max_new_tokens)
        self.assertEqual(generated.shape, (1, 3 + max_new_tokens))
        # First 3 tokens should match prompt
        torch.testing.assert_close(generated[0, :3], idx[0])


if __name__ == "__main__":
    unittest.main()
