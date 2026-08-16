import tempfile
import unittest

import torch

from nnfs.models.gpt1 import GPT1, GPT1Config
from nnfs.models.gpt2 import GPT2, GPT2Config
from nnfs.utils.model_io import load_model


class TestModelLoading(unittest.TestCase):
    def setUp(self):
        self.config = GPT1Config(
            vocab_size=32,
            block_size=16,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            dropout=0.0,
        )
        self.model = GPT1(self.config)
        self.model.eval()

        self.gpt2_config = GPT2Config(
            vocab_size=32,
            block_size=16,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            dropout=0.0,
        )
        self.gpt2_model = GPT2(self.gpt2_config)
        self.gpt2_model.eval()

    def test_load_model_roundtrip(self):
        with tempfile.TemporaryDirectory() as save_dir:
            self.model.save_pretrained(save_dir)

            loaded = load_model(save_dir, model_name="gpt1", device="cpu")

            self.assertIsInstance(loaded, GPT1)
            self.assertFalse(loaded.training)
            self.assertEqual(loaded.config.vocab_size, self.config.vocab_size)
            self.assertEqual(loaded.config.block_size, self.config.block_size)
            self.assertEqual(loaded.config.d_model, self.config.d_model)
            self.assertEqual(loaded.config.n_layers, self.config.n_layers)
            self.assertEqual(loaded.config.n_heads, self.config.n_heads)
            self.assertEqual(loaded.config.d_ff, self.config.d_ff)

            for (name, original_param), (_, loaded_param) in zip(
                self.model.named_parameters(), loaded.named_parameters()
            ):
                torch.testing.assert_close(
                    loaded_param,
                    original_param,
                    msg=f"Parameter mismatch for {name}",
                )

            x = torch.randint(0, self.config.vocab_size, (2, 8))
            with torch.no_grad():
                original_out = self.model(x)
                loaded_out = loaded(x)
            torch.testing.assert_close(loaded_out, original_out)

    def test_load_model_gpt2_roundtrip(self):
        with tempfile.TemporaryDirectory() as save_dir:
            self.gpt2_model.save_pretrained(save_dir)

            loaded = load_model(save_dir, model_name="gpt2", device="cpu")

            self.assertIsInstance(loaded, GPT2)
            self.assertFalse(loaded.training)
            self.assertTrue(hasattr(loaded, "ln_f"))
            self.assertEqual(loaded.config.vocab_size, self.gpt2_config.vocab_size)
            self.assertEqual(loaded.config.d_model, self.gpt2_config.d_model)

            for (name, original_param), (_, loaded_param) in zip(
                self.gpt2_model.named_parameters(), loaded.named_parameters()
            ):
                torch.testing.assert_close(
                    loaded_param,
                    original_param,
                    msg=f"Parameter mismatch for {name}",
                )

            x = torch.randint(0, self.gpt2_config.vocab_size, (2, 8))
            with torch.no_grad():
                original_out = self.gpt2_model(x)
                loaded_out = loaded(x)
            torch.testing.assert_close(loaded_out, original_out)

    def test_load_model_auto_infer(self):
        with tempfile.TemporaryDirectory() as save_dir:
            self.model.save_pretrained(save_dir)
            loaded = load_model(save_dir, device="cpu")
            self.assertIsInstance(loaded, GPT1)

        with tempfile.TemporaryDirectory() as save_dir:
            self.gpt2_model.save_pretrained(save_dir)
            loaded = load_model(save_dir, device="cpu")
            self.assertIsInstance(loaded, GPT2)

    def test_load_model_gpt_oss_roundtrip(self):
        from nnfs.models.gpt_oss import GptOss, GptOssConfig

        config = GptOssConfig(
            vocab_size=32,
            block_size=16,
            d_model=32,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            d_head=16,
            num_experts=4,
            top_k_experts=2,
            d_ff=32,
            dropout=0.0,
            sliding_window=8,
        )
        model = GptOss(config)
        model.eval()

        with tempfile.TemporaryDirectory() as save_dir:
            model.save_pretrained(save_dir)

            # Test explicit model_name
            loaded_explicit = load_model(save_dir, model_name="gpt_oss_moe", device="cpu")
            self.assertIsInstance(loaded_explicit, GptOss)

            # Test auto-inferred model_name=None
            loaded_auto = load_model(save_dir, device="cpu")
            self.assertIsInstance(loaded_auto, GptOss)

    def test_load_model_unknown_name(self):
        with tempfile.TemporaryDirectory() as save_dir:
            self.model.save_pretrained(save_dir)
            with self.assertRaises(KeyError):
                load_model(save_dir, model_name="unknown_model", device="cpu")


if __name__ == "__main__":
    unittest.main()
