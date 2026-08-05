import unittest
import torch
from nnfs.models.gpt1 import GPT1, GPT1Config
from nnfs.preprocessors.char_tokenizer import CharTokenizer
from nnfs.utils.causal_language_modelling import (
    CausalLanguageModelingDataset,
    CausalLanguageModelingDataLoader,
    CausalLanguageModelingTrainer,
)


class TestCausalLanguageModelling(unittest.TestCase):
    def setUp(self):
        self.texts = ["hello world", "nnfs gpt model training"]
        self.tokenizer = CharTokenizer(max_vocab_size=50)
        self.tokenizer.fit(self.texts)

        self.block_size = 16
        self.dataset = CausalLanguageModelingDataset(
            tokenizer=self.tokenizer,
            texts=self.texts,
            block_size=self.block_size,
        )

        self.config = GPT1Config(
            vocab_size=self.tokenizer.vocab_size,
            block_size=self.block_size,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            dropout=0.0,
        )
        self.model = GPT1(self.config)

    def test_dataset_item_shape(self):
        self.assertEqual(len(self.dataset), 2)
        x, y = self.dataset[0]
        self.assertEqual(x.shape, (self.block_size,))
        self.assertEqual(y.shape, (self.block_size,))
        # Target y should be x shifted by 1 position
        text_encoded = self.tokenizer.encode(
            self.texts[0], add_bos=True, add_eos=True, sequence_length=self.block_size + 1
        )
        torch.testing.assert_close(x, torch.tensor(text_encoded[:-1]))
        torch.testing.assert_close(y, torch.tensor(text_encoded[1:]))

    def test_dataloader_batching(self):
        dataloader = CausalLanguageModelingDataLoader(self.dataset, batch_size=2, shuffle=False)
        self.assertEqual(len(dataloader), 1)
        for x, y in dataloader:
            self.assertEqual(x.shape, (2, self.block_size))
            self.assertEqual(y.shape, (2, self.block_size))

    def test_trainer_overfitting(self):
        dataloader = CausalLanguageModelingDataLoader(self.dataset, batch_size=2, shuffle=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        trainer = CausalLanguageModelingTrainer(
            model=self.model,
            optimizer=optimizer,
            train_dataloader=dataloader,
            device="cpu",
        )

        initial_loss = trainer.evaluate()
        losses = trainer.train(epochs=15)
        final_loss = losses[-1]

        self.assertLess(final_loss, initial_loss)

    def test_trainer_legacy_init(self):
        # Test backward-compatible initialization: Trainer(model, tokenizer, dataset, batch_size)
        trainer = CausalLanguageModelingTrainer(
            self.model,
            self.tokenizer,
            self.dataset,
            batch_size=2,
        )
        loss, step = trainer.train_epoch()
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        self.assertGreater(step, 0)


if __name__ == "__main__":
    unittest.main()
