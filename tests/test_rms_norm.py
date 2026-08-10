import unittest
import torch

from nnfs.layers import RMSNorm


class TestRMSNorm(unittest.TestCase):
    def test_output_shape(self):
        norm = RMSNorm(d_model=64)
        x = torch.randn(4, 16, 64)
        out = norm(x)
        self.assertEqual(out.shape, (4, 16, 64))

    def test_rms_normalization(self):
        d_model = 128
        norm = RMSNorm(d_model=d_model, eps=1e-8)
        # Initialize gamma to 1
        with torch.no_grad():
            norm.gamma.fill_(1.0)
        x = torch.randn(2, 8, d_model) * 5.0  # arbitrary scale
        out = norm(x)

        # RMS of each normalized output vector should be approximately 1.0
        rms_out = torch.sqrt(torch.mean(out ** 2, dim=-1))
        torch.testing.assert_close(rms_out, torch.ones_like(rms_out), rtol=1e-4, atol=1e-4)

    def test_gradient_flow(self):
        norm = RMSNorm(d_model=32)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.gamma.grad)
        self.assertFalse(torch.isnan(norm.gamma.grad).any())
        self.assertFalse(torch.isnan(x.grad).any())


if __name__ == "__main__":
    unittest.main()
