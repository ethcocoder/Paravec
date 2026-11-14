import unittest

import numpy as np

from paravec.analyzer import analyze_spectrum


class AnalyzerTests(unittest.TestCase):
    def test_infeasible_budgets_are_reported_for_low_dimension(self):
        rng = np.random.default_rng(4)
        report = analyze_spectrum(rng.normal(size=(20, 2)), bit_budgets=(8, 24))

        self.assertTrue(report["allocations"][0]["feasible"])
        self.assertFalse(report["allocations"][1]["feasible"])

    def test_spectrum_report_accounts_for_packed_index_overheads(self):
        rng = np.random.default_rng(8)
        report = analyze_spectrum(rng.normal(size=(100, 8)), bit_budgets=(32,))
        allocation = report["allocations"][0]

        self.assertEqual(allocation["packed_code_bytes"], 400)
        self.assertEqual(allocation["error_bound_bytes"], 800)
        self.assertGreater(allocation["estimated_index_bytes_per_vector"], 8.0)
        self.assertGreater(allocation["estimated_compression_ratio_vs_float32"], 0.0)


if __name__ == "__main__":
    unittest.main()