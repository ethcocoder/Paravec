from pathlib import Path
import tempfile
import unittest

import numpy as np

from paravec.search import PackedSpectralIndex, SpectralIndex


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

    def test_packed_index_reports_bounds_and_uses_less_numeric_memory(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(256, 16)) * np.geomspace(3.0, 0.2, 16)
        query = rng.normal(size=16)
        index = PackedSpectralIndex(vectors, bit_budget_bits_per_vector=64)

        result = index.search(query, k=10)
        exact_distances = np.sum((vectors - query) ** 2, axis=1)

        self.assertFalse(result.exact)
        self.assertGreater(index.compression_ratio, 1.0)
        self.assertTrue(np.all(result.lower_bounds <= exact_distances[result.indices]))
        self.assertTrue(np.all(exact_distances[result.indices] <= result.upper_bounds + 1e-10))

    def test_packed_exact_rerank_matches_brute_force(self):
        rng = np.random.default_rng(23)
        vectors = rng.normal(size=(128, 12)) * np.geomspace(2.0, 0.1, 12)
        query = rng.normal(size=12)
        index = PackedSpectralIndex(vectors, bit_budget_bits_per_vector=48, store_original=True)

        result = index.search(query, k=7, exact_rerank=True)
        expected = np.argsort(np.sum((vectors - query) ** 2, axis=1), kind="stable")[:7]

        self.assertTrue(result.exact)
        np.testing.assert_array_equal(result.indices, expected)
        self.assertLess(result.shortlist_size, len(vectors))

    def test_packed_search_is_invariant_to_decode_block_size(self):
        rng = np.random.default_rng(31)
        vectors = rng.normal(size=(137, 11)) * np.geomspace(2.0, 0.1, 11)
        query = rng.normal(size=11)
        index = PackedSpectralIndex(vectors, bit_budget_bits_per_vector=33, store_original=True)
        expected = np.argsort(np.sum((vectors - query) ** 2, axis=1), kind="stable")[:8]
        reference = index.search(query, k=8, exact_rerank=True, block_size=137)

        for block_size in (1, 7, 32, 64):
            with self.subTest(block_size=block_size):
                result = index.search(query, k=8, exact_rerank=True, block_size=block_size)
                np.testing.assert_array_equal(result.indices, expected)
                np.testing.assert_array_equal(result.indices, reference.indices)
                np.testing.assert_allclose(result.squared_distances, reference.squared_distances)

    def test_packed_search_rejects_invalid_block_size(self):
        index = PackedSpectralIndex(np.eye(4), bit_budget_bits_per_vector=16)
        with self.assertRaisesRegex(ValueError, "block_size"):
            index.search(np.zeros(4), k=2, block_size=0)
        with self.assertRaisesRegex(ValueError, "working-memory"):
            index.search(np.zeros(4), k=2, block_size=1_000_001)

    def test_packed_index_save_load_preserves_queries(self):
        rng = np.random.default_rng(17)
        vectors = rng.normal(size=(96, 10)) * np.geomspace(2.0, 0.2, 10)
        query = rng.normal(size=10)
        index = PackedSpectralIndex(vectors, bit_budget_bits_per_vector=40, store_original=True)
        before = index.search(query, k=6, exact_rerank=True)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.npz"
            index.save(path)
            loaded = PackedSpectralIndex.load(path)
            after = loaded.search(query, k=6, exact_rerank=True)

        np.testing.assert_array_equal(after.indices, before.indices)
        np.testing.assert_allclose(after.squared_distances, before.squared_distances)
        self.assertEqual(loaded.numeric_storage_bytes, index.numeric_storage_bytes)


if __name__ == "__main__":
    unittest.main()