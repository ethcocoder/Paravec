import unittest

import numpy as np

from paravec.evaluation import evaluate_retrieval


class EvaluationTests(unittest.TestCase):
    def test_exact_rerank_recall_is_one_and_memory_is_reported(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(100, 8)) * np.geomspace(3.0, 0.2, 8)

        report = evaluate_retrieval(
            vectors,
            bit_budgets=(16, 32),
            query_count=10,
            k=5,
            seed=5,
        )

        self.assertEqual(len(report["results"]), 2)
        for result in report["results"]:
            self.assertTrue(result["feasible"])
            self.assertEqual(result["exact_rerank_recall_at_k_mean"], 1.0)
            scalar = result["uniform_scalar"]
            self.assertTrue(scalar["feasible"])
            self.assertEqual(result["uniform_scalar"]["exact_rerank_recall_at_k_mean"], 1.0)
            self.assertEqual(scalar["code_bytes_mean"], result["code_bytes_mean"])
            self.assertGreater(result["packed_numeric_storage_bytes_mean"], 0)
            self.assertGreater(result["rerank_bytes_per_vector_mean"], result["packed_bytes_per_vector_mean"])

    def test_invalid_budget_values_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval(np.ones((10, 4)), bit_budgets=())
        with self.assertRaisesRegex(ValueError, "integers"):
            evaluate_retrieval(np.ones((10, 4)), bit_budgets=(8.5,))

    def test_repeated_splits_are_deterministic_and_exact_rerank_stays_exact(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(80, 8)) * np.geomspace(2.0, 0.2, 8)
        arguments = {
            "bit_budgets": (16, 32),
            "query_count": 5,
            "k": 4,
            "repeats": 3,
            "seed": 13,
        }

        first = evaluate_retrieval(vectors, **arguments)
        second = evaluate_retrieval(vectors, **arguments)

        self.assertEqual(first["seeds"], [13, 14, 15])
        for first_budget, second_budget in zip(first["results"], second["results"]):
            self.assertEqual(first_budget["runs"], second_budget["runs"])
            self.assertEqual(first_budget["exact_rerank_recall_at_k_mean"], 1.0)
            self.assertEqual(first_budget["exact_rerank_recall_at_k_min"], 1.0)


if __name__ == "__main__":
    unittest.main()