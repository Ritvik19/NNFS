import json
import tempfile
import unittest
import torch

from nnfs.models.mixtral_moe import MixtralMoE, MixtralMoEConfig
from src.router_analysis import RouterTracker, print_analysis_report


class TestRouterAnalysis(unittest.TestCase):
    def setUp(self):
        self.config = MixtralMoEConfig(
            vocab_size=100,
            block_size=16,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            num_experts=4,
            top_k_experts=2,
            d_ff=64,
            dropout=0.0,
            rope_theta=10000.0,
        )
        self.model = MixtralMoE(self.config)

    def test_router_tracker_registration(self):
        tracker = RouterTracker(self.model)
        self.assertEqual(len(tracker.router_data), 2)
        for layer_name, data in tracker.router_data.items():
            self.assertIn("blocks", layer_name)
            self.assertEqual(data["num_experts"], 4)
        tracker.remove_hooks()

    def test_router_tracker_token_capture(self):
        tracker = RouterTracker(self.model)
        self.model.eval()

        batch_size, seq_len = 2, 8
        x = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            _ = self.model(x)

        metrics = tracker.compute_metrics()
        tracker.remove_hooks()

        self.assertIn("global_summary", metrics)
        # 2 layers, B=2, T=8, K=2 -> total routing slots per layer = 2 * 8 * 2 = 32
        for layer_name in ["blocks.0.moe.router", "blocks.1.moe.router"]:
            self.assertIn(layer_name, metrics)
            self.assertEqual(metrics[layer_name]["total_token_slots"], 32)
            self.assertEqual(sum(metrics[layer_name]["counts"]), 32)
            self.assertAlmostEqual(sum(metrics[layer_name]["percentages"]), 100.0, delta=0.5)

        # Global summary total slots = 32 * 2 = 64
        self.assertEqual(metrics["global_summary"]["total_token_slots"], 64)
        self.assertEqual(sum(metrics["global_summary"]["counts"]), 64)

    def test_print_analysis_report(self):
        tracker = RouterTracker(self.model)
        self.model.eval()
        x = torch.randint(0, 100, (2, 4))
        with torch.no_grad():
            _ = self.model(x)

        metrics = tracker.compute_metrics()
        tracker.remove_hooks()

        # Ensure print_analysis_report runs without error
        try:
            print_analysis_report(metrics)
        except Exception as e:
            self.fail(f"print_analysis_report raised exception: {e}")

    def test_json_serialization(self):
        tracker = RouterTracker(self.model)
        self.model.eval()
        x = torch.randint(0, 100, (2, 4))
        with torch.no_grad():
            _ = self.model(x)

        metrics = tracker.compute_metrics()
        tracker.remove_hooks()

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            json.dump(metrics, f)
            temp_path = f.name

        with open(temp_path, "r") as f:
            loaded_metrics = json.load(f)

        self.assertEqual(metrics.keys(), loaded_metrics.keys())

    def test_load_validation_texts_all_samples(self):
        from src.router_analysis import load_validation_texts

        # Mock synthetic fallback by giving dummy path
        texts_sampled = load_validation_texts(data_path="invalid_path", num_samples=10)
        self.assertEqual(len(texts_sampled), 10)

        texts_all_neg = load_validation_texts(data_path="invalid_path", num_samples=-1)
        self.assertEqual(len(texts_all_neg), 100)

        texts_all_zero = load_validation_texts(data_path="invalid_path", num_samples=0)
        self.assertEqual(len(texts_all_zero), 100)


if __name__ == "__main__":
    unittest.main()
