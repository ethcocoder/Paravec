import unittest

import numpy as np

from paravec.validation import validate_dataset


class ValidationTests(unittest.TestCase):
    def test_validation_checks_isometry_search_and_deterministic_bounds(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(60, 8)) * np.geomspace(3.0, 0.2, 8)

        report = validate_dataset(vectors, query_count=8, k=5, bit_budget=24, seed=13)

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["isometry_passed"])
        self.assertTrue(report["exact_staged_search_matches_brute_force"])
        self.assertTrue(report["deterministic_bound_passed"])
        self.assertEqual(report["deterministic_bound_coverage"], 1.0)
        self.assertGreater(report["deterministic_bound_pair_count"], 0)

    def test_validation_requires_heldout_data(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            validate_dataset(np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()