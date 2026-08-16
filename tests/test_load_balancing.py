import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nnfs.losses import LoadBalancingLoss
from nnfs.layers.sparse_moe import TopKRouter, SparseMoE
from nnfs.models.mixtral_moe import MixtralMoE, MixtralMoEConfig
from nnfs.preprocessors.char_tokenizer import CharTokenizer
from nnfs.utils.causal_language_modelling import (
    CausalLanguageModelingDataset,
    CausalLanguageModelingDataLoader,
    CausalLanguageModelingTrainer,
)


class TestLoadBalancingLoss(unittest.TestCase):
    def setUp(self):
        self.lb_loss_fn = LoadBalancingLoss()

    def test_single_layer_balanced(self):
        # 4 tokens, 4 experts, top-1 routing (1 token assigned to each expert)
        # Uniform router logits giving equal softmax probabilities (P_i = 0.25)
        num_experts = 4
        router_logits = torch.ones(4, num_experts)
        top_k_indices = torch.tensor([[0], [1], [2], [3]])

        loss = self.lb_loss_fn((router_logits, top_k_indices))
        # f_i = 0.25 for all i, P_i = 0.25 for all i.
        # N * sum(f_i * P_i) = 4 * (4 * 0.25 * 0.25) = 1.0
        self.assertAlmostEqual(loss.item(), 1.0, places=4)

    def test_imbalanced_loss_higher(self):
        num_experts = 4
        router_logits_balanced = torch.ones(8, num_experts)
        top_k_balanced = torch.tensor([[0], [1], [2], [3], [0], [1], [2], [3]])

        # All tokens assigned to expert 0 with high confidence for expert 0
        router_logits_imbalanced = torch.tensor(
            [[10.0, 0.0, 0.0, 0.0] for _ in range(8)]
        )
        top_k_imbalanced = torch.zeros((8, 1), dtype=torch.long)

        loss_bal = self.lb_loss_fn((router_logits_balanced, top_k_balanced))
        loss_imbal = self.lb_loss_fn((router_logits_imbalanced, top_k_imbalanced))

        self.assertGreater(loss_imbal.item(), loss_bal.item())

    def test_multiple_layers_list(self):
        num_experts = 4
        layer1_logits = torch.randn(2, 8, num_experts, requires_grad=True)
        layer1_indices = torch.randint(0, num_experts, (2, 8, 2))

        layer2_logits = torch.randn(2, 8, num_experts, requires_grad=True)
        layer2_indices = torch.randint(0, num_experts, (2, 8, 2))

        router_outputs = [
            (layer1_logits, layer1_indices),
            (layer2_logits, layer2_indices),
        ]

        loss = self.lb_loss_fn(router_outputs)
        self.assertTrue(loss.requires_grad)
        loss.backward()

        self.assertIsNotNone(layer1_logits.grad)
        self.assertIsNotNone(layer2_logits.grad)


class TestTopKRouterStateCaching(unittest.TestCase):
    def test_router_caching(self):
        router = TopKRouter(d_model=16, num_experts=4, top_k=2)
        x = torch.randn(2, 5, 16)

        self.assertIsNone(router.last_router_logits)
        self.assertIsNone(router.last_top_k_indices)

        weights, indices = router(x)

        self.assertIsNotNone(router.last_router_logits)
        self.assertIsNotNone(router.last_top_k_indices)
        self.assertEqual(router.last_router_logits.shape, (2, 5, 4))
        self.assertEqual(router.last_top_k_indices.shape, (2, 5, 2))
        torch.testing.assert_close(indices, router.last_top_k_indices)


class TestMixtralMoERouterOutputs(unittest.TestCase):
    def test_get_router_outputs(self):
        config = MixtralMoEConfig(
            vocab_size=64,
            d_model=32,
            n_layers=2,
            n_heads=2,
            num_experts=4,
            top_k_experts=2,
            block_size=16,
        )
        model = MixtralMoE(config)
        x = torch.randint(0, 64, (2, 8))

        _ = model(x)
        router_outputs = model.get_router_outputs()

        self.assertEqual(len(router_outputs), config.n_layers)
        for logits, indices in router_outputs:
            self.assertEqual(logits.shape, (2, 8, 4))
            self.assertEqual(indices.shape, (2, 8, 2))


class TestTrainerLoadBalancingIntegration(unittest.TestCase):
    def test_trainer_with_lb_loss(self):
        config = MixtralMoEConfig(
            vocab_size=32,
            d_model=16,
            n_layers=2,
            n_heads=2,
            num_experts=4,
            top_k_experts=2,
            block_size=16,
        )
        model = MixtralMoE(config)

        texts = ["hello world story", "mixture of experts training test"]
        tokenizer = CharTokenizer(max_vocab_size=32)
        tokenizer.fit(texts)

        dataset = CausalLanguageModelingDataset(tokenizer, texts, block_size=16)
        dataloader = CausalLanguageModelingDataLoader(dataset, batch_size=2)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = CausalLanguageModelingTrainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=dataloader,
            load_balancing_coef=0.05,
        )

        step_records = []

        def callback(step, loss, ce_loss=0.0, lb_loss=0.0):
            step_records.append((step, loss, ce_loss, lb_loss))

        avg_loss, steps = trainer.train_epoch(on_step=callback)

        self.assertGreater(steps, 0)
        self.assertGreater(avg_loss, 0.0)
        self.assertEqual(len(step_records), steps)
        for _, loss, ce_loss, lb_loss in step_records:
            self.assertGreater(ce_loss, 0.0)
            self.assertGreater(lb_loss, 0.0)
            self.assertAlmostEqual(loss, ce_loss + 0.05 * lb_loss, places=4)


if __name__ == "__main__":
    unittest.main()
