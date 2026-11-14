import unittest

import numpy as np

from paravec.analyzer import analyze_spectrum


class AnalyzerTests(unittest.TestCase):
    def test_infeasible_budgets_are_reported_for_low_dimension(self):
        rng = np.random.default_rng(4)
        report = analyze_spectrum(rng.normal(size=(20, 2)), bit_budgets=(8, 24))

        self.assertTrue(report["allocations"][0]["feasible"])
        self.assertFalse(report["allocations"][1]["feasible"])


if __name__ == "__main__":
    unittest.main()