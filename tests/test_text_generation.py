import unittest
import torch
from nnfs.models.gpt1 import GPT1, GPT1Config
from nnfs.preprocessors.char_tokenizer import CharTokenizer
from nnfs.utils.text_generation import generate, generate_text


class TestTextGeneration(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizer(max_vocab_size=30)
        self.tokenizer.fit(["hello world text generation test"])

        self.config = GPT1Config(
            vocab_size=self.tokenizer.vocab_size,
            block_size=16,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            dropout=0.0,
        )
        self.model = GPT1(self.config)

    def test_generate_tensor(self):
        idx = torch.tensor([[1, 2, 3]])
        out_idx = generate(self.model, idx, max_new_tokens=5, temperature=1.0)
        self.assertEqual(out_idx.shape, (1, 8))
        torch.testing.assert_close(out_idx[0, :3], idx[0])

    def test_generate_text_basic(self):
        prompt = "hello"
        out_text = generate_text(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            temperature=1.0,
        )
        self.assertIsInstance(out_text, str)
        # Check that output starts with prompt when special tokens are skipped
        self.assertTrue(out_text.startswith(prompt))

    def test_generate_text_top_k_and_top_p(self):
        prompt = "test"
        out_text = generate_text(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            temperature=0.8,
            top_k=5,
            top_p=0.9,
        )
        self.assertIsInstance(out_text, str)

    def test_generate_text_greedy(self):
        prompt = "world"
        out_text1 = generate_text(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_new_tokens=8,
            temperature=0.0,
        )
        out_text2 = generate_text(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_new_tokens=8,
            temperature=0.0,
        )
        # Greedy decoding should be deterministic
        self.assertEqual(out_text1, out_text2)


if __name__ == "__main__":
    unittest.main()
