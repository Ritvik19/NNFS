import os
import tempfile
import unittest
from nnfs.preprocessors.char_tokenizer import CharTokenizer


class TestCharTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizer(max_vocab_size=10)

    def test_initial_state(self):
        self.assertEqual(len(self.tokenizer.vocab), 4)
        self.assertEqual(self.tokenizer.vocab_size, 4)
        self.assertEqual(len(self.tokenizer), 4)
        self.assertIn("<pad>", self.tokenizer.vocab)
        self.assertIn("<unk>", self.tokenizer.vocab)
        self.assertIn("<bos>", self.tokenizer.vocab)
        self.assertIn("<eos>", self.tokenizer.vocab)

    def test_fit(self):
        texts = ["hello world!", "testing tokenizer."]
        self.tokenizer.fit(texts)

        # max_vocab_size = 10 (4 special tokens + top 6 characters)
        self.assertEqual(self.tokenizer.vocab_size, 10)
        self.assertEqual(len(self.tokenizer), 10)

    def test_encode_unpadded(self):
        self.tokenizer.fit(["abc"])
        # 'a', 'b', 'c' are present in vocab
        encoded = self.tokenizer.encode("abc", add_bos=True, add_eos=True, sequence_length=None)
        # Expected: [<bos>, a, b, c, <eos>]
        self.assertEqual(len(encoded), 5)
        self.assertEqual(encoded[0], self.tokenizer.char2idx["<bos>"])
        self.assertEqual(encoded[-1], self.tokenizer.char2idx["<eos>"])

    def test_encode_padding(self):
        self.tokenizer.fit(["abc"])
        encoded = self.tokenizer.encode("abc", add_bos=True, add_eos=True, sequence_length=10)
        # Sequence padded to length 10
        self.assertEqual(len(encoded), 10)
        # Left padded with <pad> (index 0)
        self.assertEqual(encoded[:5], [self.tokenizer.char2idx["<pad>"]] * 5)
        self.assertEqual(encoded[5], self.tokenizer.char2idx["<bos>"])
        self.assertEqual(encoded[-1], self.tokenizer.char2idx["<eos>"])

    def test_encode_truncation(self):
        self.tokenizer.fit(["abcdefghijklmnopqrstuvwxyz"])
        # String length + BOS + EOS = 28 tokens
        encoded = self.tokenizer.encode(
            "abcdefghijklmnopqrstuvwxyz", add_bos=True, add_eos=True, sequence_length=10
        )
        # Must be truncated to EXACTLY sequence_length (10)
        self.assertEqual(len(encoded), 10)
        # Keeps first 10 tokens: <bos> + 'a' .. 'h'
        self.assertEqual(encoded[0], self.tokenizer.char2idx["<bos>"])
        self.assertEqual(encoded[1], self.tokenizer.char2idx["a"])

    def test_unknown_character(self):
        self.tokenizer.fit(["abc"])
        # 'z' is unknown character
        encoded = self.tokenizer.encode("z", add_bos=False, add_eos=False, sequence_length=None)
        self.assertEqual(encoded, [self.tokenizer.char2idx["<unk>"]])

    def test_decode(self):
        self.tokenizer.fit(["hello"])
        encoded = self.tokenizer.encode("hello", add_bos=True, add_eos=True, sequence_length=None)
        
        # Raw decode includes special tokens
        raw_decoded = self.tokenizer.decode(encoded, skip_special_tokens=False)
        self.assertEqual(raw_decoded, "<bos>hello<eos>")

        # Clean decode skips special tokens
        clean_decoded = self.tokenizer.decode(encoded, skip_special_tokens=True)
        self.assertEqual(clean_decoded, "hello")

    def test_save_and_load(self):
        self.tokenizer.fit(["save load test"])
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tokenizer.json")
            self.tokenizer.save(filepath)

            loaded_tokenizer = CharTokenizer.load(filepath)
            self.assertEqual(self.tokenizer.vocab, loaded_tokenizer.vocab)
            self.assertEqual(self.tokenizer.vocab_size, loaded_tokenizer.vocab_size)
            self.assertEqual(self.tokenizer.max_vocab_size, loaded_tokenizer.max_vocab_size)
            
            # Encoded outputs match
            text = "save test"
            enc1 = self.tokenizer.encode(text)
            enc2 = loaded_tokenizer.encode(text)
            self.assertEqual(enc1, enc2)


if __name__ == "__main__":
    unittest.main()
