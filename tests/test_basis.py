import unittest

import numpy as np

from paravec.basis import effective_dimensions, fit_basis, participation_ratio


class BasisTests(unittest.TestCase):
    def test_pca_transform_preserves_pairwise_distances(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(80, 12))
        basis = fit_basis(vectors)
        coefficients = basis.transform(vectors)
        query = rng.normal(size=12)
        query_coefficients = basis.transform(query[None, :])[0]

        np.testing.assert_allclose(
            np.sum((vectors - query) ** 2, axis=1),
            np.sum((coefficients - query_coefficients) ** 2, axis=1),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_exact_prefix_distances_are_lower_bounds(self):
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(30, 8))
        query = rng.normal(size=8)
        basis = fit_basis(vectors)
        coefficients = basis.transform(vectors)
        query_coefficients = basis.transform(query[None, :])[0]
        partial = np.sum((coefficients[:, :3] - query_coefficients[:3]) ** 2, axis=1)
        exact = np.sum((vectors - query) ** 2, axis=1)

        self.assertTrue(np.all(partial <= exact + 1e-10))

    def test_spectrum_summary_handles_flat_and_zero_spectra(self):
        self.assertEqual(effective_dimensions([1, 1, 1, 1])[0.9], 4)
        self.assertEqual(effective_dimensions([0, 0, 0])[0.99], 0)
        self.assertAlmostEqual(participation_ratio([4, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()