import unittest

import numpy as np

from paravec.alloc import allocate_bits, predicted_distortion
from paravec.quant import quantize, squared_distance_error_bound


class QuantizationTests(unittest.TestCase):
    def test_allocator_spends_budget_and_prioritizes_variance(self):
        eigenvalues = np.array([8.0, 4.0, 2.0, 1.0])

        bits = allocate_bits(eigenvalues, budget_bits=20, max_bits=8)

        self.assertEqual(int(bits.sum()), 20)
        self.assertTrue(np.all(bits[:-1] >= bits[1:]))
        self.assertLess(predicted_distortion(eigenvalues, bits), eigenvalues.sum())

    def test_quantization_distance_error_respects_deterministic_bound(self):
        rng = np.random.default_rng(12)
        eigenvalues = np.array([3.0, 1.5, 0.5, 0.1])
        bits = np.array([5, 4, 3, 2], dtype=np.int16)
        coefficients = rng.normal(size=(100, 4)) * np.sqrt(eigenvalues)
        query = rng.normal(size=4)
        quantized = quantize(coefficients, eigenvalues, bits)
        estimated = np.sum((query - quantized.reconstructed) ** 2, axis=1)
        exact = np.sum((query - coefficients) ** 2, axis=1)
        bounds = np.array([
            squared_distance_error_bound(query, row, error)
            for row, error in zip(quantized.reconstructed, quantized.error_norm)
        ])

        self.assertTrue(np.all(np.abs(exact - estimated) <= bounds + 1e-10))

    def test_allocator_rejects_unrepresentable_budget(self):
        with self.assertRaises(ValueError):
            allocate_bits([1.0, 0.0], budget_bits=9, max_bits=8)


if __name__ == "__main__":
    unittest.main()