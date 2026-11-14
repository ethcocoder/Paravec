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
            self.assertGreater(result["packed_numeric_storage_bytes"], 0)
            self.assertGreater(result["rerank_storage_bytes"], result["packed_numeric_storage_bytes"])

    def test_invalid_budget_values_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval(np.ones((10, 4)), bit_budgets=())
        with self.assertRaisesRegex(ValueError, "integers"):
            evaluate_retrieval(np.ones((10, 4)), bit_budgets=(8.5,))


if __name__ == "__main__":
    unittest.main()