import json
from collections import Counter
from typing import Optional
from tqdm.auto import tqdm

class CharTokenizer:
    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(self, max_vocab_size: int = 192):
        self.max_vocab_size = max_vocab_size
        self.reset_tokenizer()

    def reset_tokenizer(self):
        self.vocab = list(self.SPECIAL_TOKENS)
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def fit(self, texts: list[str]):
        self.reset_tokenizer()
        counter = Counter()
        for text in tqdm(texts, desc="Counting characters"):
            counter.update(text)
        most_common = counter.most_common(max(0, self.max_vocab_size - len(self.vocab)))
        self.vocab.extend([char for char, _ in most_common])
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        sequence_length: Optional[int] = None,
    ) -> list[int]:
        encoded = [self.char2idx.get(char, self.char2idx['<unk>']) for char in text]
        if add_bos:
            encoded.insert(0, self.char2idx['<bos>'])
        if add_eos:
            encoded.append(self.char2idx['<eos>'])
        if sequence_length is not None:
            if len(encoded) < sequence_length:
                # pad the sequence to the left
                encoded = [self.char2idx['<pad>']] * (sequence_length - len(encoded)) + encoded
            elif len(encoded) > sequence_length:
                # truncate the sequence to the right (keep first sequence_length tokens)
                encoded = encoded[:sequence_length]
        return encoded

    def decode(self, encoded: list[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            special_indices = {self.char2idx[tok] for tok in self.SPECIAL_TOKENS if tok in self.char2idx}
            encoded = [idx for idx in encoded if idx not in special_indices]
        return ''.join([self.idx2char.get(idx, '<unk>') for idx in encoded])

    def save(self, filepath: str):
        data = {
            "max_vocab_size": self.max_vocab_size,
            "vocab": self.vocab,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(max_vocab_size=data.get("max_vocab_size", 192))
        tokenizer.vocab = data["vocab"]
        tokenizer.char2idx = {char: idx for idx, char in enumerate(tokenizer.vocab)}
        tokenizer.idx2char = {idx: char for idx, char in enumerate(tokenizer.vocab)}
        return tokenizer
