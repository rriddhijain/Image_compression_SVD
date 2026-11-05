"""
Unit Tests & Validation Suite.
Person B: Hardware Systems Lead.

Validates:
1. Output tensor shapes for standard vs. decomposed layers.
2. Numerical equivalence at full rank k = min(M, N).
3. Monotonic decrease of Frobenius approximation error as rank k increases.
4. Correctness of analytical FLOP calculation and breakeven rank.
"""

import unittest
import torch
import torch.nn as nn
from models.decomposed import DecomposedLinear, decompose_model_linear_layers
from models.baseline import BaselineMLP
from profiling.metrics import (
    compute_linear_flops,
    compute_decomposed_flops,
    compute_breakeven_rank,
    calculate_arithmetic_intensity
)


class TestSVDDecompositionAndProfiling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 16
        self.in_features = 128
        self.out_features = 64
        self.full_rank = min(self.in_features, self.out_features)
        
        self.orig_linear = nn.Linear(self.in_features, self.out_features, bias=True)
        self.x = torch.randn(self.batch_size, self.in_features)

    def test_full_rank_numerical_equivalence(self):
        """Full-rank SVD reconstruction must match original forward pass within float precision."""
        decomposed_full = DecomposedLinear.from_linear(
            self.orig_linear, rank=self.full_rank, svd_scheme="symmetric"
        )
        
        with torch.no_grad():
            y_orig = self.orig_linear(self.x)
            y_decomp = decomposed_full(self.x)

        max_abs_diff = torch.max(torch.abs(y_orig - y_decomp)).item()
        rel_diff = (torch.norm(y_orig - y_decomp) / torch.norm(y_orig)).item()
        
        self.assertLess(max_abs_diff, 1e-4, f"Full rank diff too high: {max_abs_diff}")
        self.assertLess(rel_diff, 1e-4, f"Relative diff too high: {rel_diff}")

    def test_output_shapes(self):
        """Ensure shape preservation for decomposed layers across arbitrary ranks."""
        for k in [4, 8, 16, 32]:
            decomp = DecomposedLinear.from_linear(self.orig_linear, rank=k)
            y = decomp(self.x)
            self.assertEqual(y.shape, (self.batch_size, self.out_features))

    def test_monotonic_error_decay(self):
        """Reconstruction error must monotonically decrease as rank k increases."""
        errors = []
        ranks = [4, 8, 16, 32, 64]
        y_orig = self.orig_linear(self.x)

        for k in ranks:
            decomp = DecomposedLinear.from_linear(self.orig_linear, rank=k)
            y_k = decomp(self.x)
            err = torch.norm(y_orig - y_k).item()
            errors.append(err)

        for i in range(len(errors) - 1):
            self.assertGreaterEqual(errors[i] + 1e-5, errors[i+1], "Reconstruction error must decrease or stay equal")

    def test_breakeven_rank_formula(self):
        """At breakeven rank K*, FLOPs of decomposed layer equals standard layer."""
        m, n = 1024, 512
        k_star = compute_breakeven_rank(m, n)
        
        # Verify k_star formula: (1024 * 512) / (1024 + 512) = 524288 / 1536 = 341.333
        expected_k_star = (1024 * 512) / (1024 + 512)
        self.assertAlmostEqual(k_star, expected_k_star, places=4)

        # Flops comparison (ignoring single bias addition)
        flops_standard = 2 * 1 * m * n
        flops_decomp_at_kstar = 2 * 1 * k_star * (m + n)
        self.assertAlmostEqual(flops_standard, flops_decomp_at_kstar, places=2)

    def test_end_to_end_model_swapping(self):
        """Verify model-wide layer swapping replaces target linear layers and preserves output shape."""
        model = BaselineMLP(in_features=3072, num_classes=10)
        x_in = torch.randn(4, 3072)
        y_orig = model(x_in)
        self.assertEqual(y_orig.shape, (4, 10))

        decomposed_model = decompose_model_linear_layers(model, rank=32)
        y_decomp = decomposed_model(x_in)
        self.assertEqual(y_decomp.shape, (4, 10))

        # Check that fc1, fc2, fc3 were swapped to DecomposedLinear
        self.assertIsInstance(decomposed_model.fc1, DecomposedLinear)
        self.assertIsInstance(decomposed_model.fc2, DecomposedLinear)
        self.assertIsInstance(decomposed_model.fc3, DecomposedLinear)
        # fc4 should remain standard linear because out_features=10 <= 10
        self.assertIsInstance(decomposed_model.fc4, nn.Linear)


if __name__ == "__main__":
    unittest.main()
