import unittest

import torch
import torch.nn as nn

from nnfs.activations import SwiGLU
from nnfs.layers import GroupedQueryAttention, RMSNorm, SwiGLUMLP, apply_rotary_pos_emb
from nnfs.models.palm2 import PaLM2, PaLM2Config
from nnfs.modules import PaLM2TransformerBlock
from nnfs.utils.model_io import build_model
from nnfs.utils.text_generation import generate


class TestPaLM2Components(unittest.TestCase):
    def test_swiglu_activation(self):
        swiglu = SwiGLU()
        gate = torch.randn(2, 4, 16)
        up = torch.randn(2, 4, 16)
        out = swiglu(gate, up)
        self.assertEqual(out.shape, (2, 4, 16))

    def test_swiglu_mlp(self):
        mlp = SwiGLUMLP(d_model=64, d_ff=128, dropout=0.1, bias=False)
        x = torch.randn(2, 8, 64)
        out = mlp(x)
        self.assertEqual(out.shape, (2, 8, 64))
        self.assertIsNone(mlp.w_gate.bias)
        self.assertIsNone(mlp.w_up.bias)
        self.assertIsNone(mlp.w_down.bias)

    def test_rmsnorm(self):
        norm = RMSNorm(d_model=64, eps=1e-5)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_grouped_query_attention_shape(self):
        gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout=0.1, bias=False)
        x = torch.randn(2, 8, 64)
        out = gqa(x)
        self.assertEqual(out.shape, (2, 8, 64))
        self.assertIsNone(gqa.q_proj.bias)
        self.assertIsNone(gqa.k_proj.bias)
        self.assertIsNone(gqa.v_proj.bias)
        self.assertIsNone(gqa.out_proj.bias)
        self.assertEqual(gqa.k_proj.weights.shape, (64, 32))  # 2 heads of dim 16

    def test_palm2_transformer_block(self):
        block = PaLM2TransformerBlock(d_model=64, n_heads=4, n_kv_heads=2, d_ff=128, dropout=0.1)
        x = torch.randn(2, 8, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 8, 64))


class TestPaLM2Model(unittest.TestCase):
    def setUp(self):
        self.config = PaLM2Config(
            vocab_size=100,
            block_size=32,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            dropout=0.1,
            eps=1e-5,
        )
        self.model = PaLM2(self.config)

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
        model = build_model("configs/palm2_config.yaml")
        self.assertIsInstance(model, PaLM2)
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
