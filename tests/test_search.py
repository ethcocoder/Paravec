import unittest

import numpy as np

from paravec.search import SpectralIndex


class SearchTests(unittest.TestCase):
    def test_staged_search_matches_exact_brute_force_top_k(self):
        rng = np.random.default_rng(23)
        vectors = rng.normal(size=(500, 24)) * np.geomspace(4.0, 0.1, 24)
        query = rng.normal(size=24)
        index = SpectralIndex(vectors)

        result = index.search(query, k=10, stages=[4, 8, 16, 24])
        expected = np.argsort(np.sum((vectors - query) ** 2, axis=1))[:10]

        np.testing.assert_array_equal(result.indices, expected)
        self.assertLess(result.coefficient_reads, result.total_coefficient_reads)
        self.assertGreaterEqual(result.survivors_by_stage[-1], len(result.indices))
        self.assertLessEqual(result.survivors_by_stage[-1], result.survivors_by_stage[0])

    def test_invalid_stages_are_rejected(self):
        index = SpectralIndex(np.arange(40, dtype=np.float64).reshape(10, 4))

        with self.assertRaises(ValueError):
            index.search(np.zeros(4), stages=[2, 3])


if __name__ == "__main__":
    unittest.main()