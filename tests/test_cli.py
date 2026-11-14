import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paravec.cli import main


class CliTests(unittest.TestCase):
    def test_analyze_writes_a_json_spectrum_report(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "vectors.npy"
            output_path = Path(directory) / "report.json"
            np.save(input_path, np.arange(60, dtype=np.float64).reshape(20, 3))

            result = main(["analyze", str(input_path), "--output", str(output_path)])
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(report["vector_count"], 20)
        self.assertEqual(report["dimension"], 3)

    def test_validate_writes_validation_report(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "vectors.npy"
            output_path = Path(directory) / "validation.json"
            np.save(input_path, np.random.default_rng(5).normal(size=(30, 6)))

            result = main(["validate", str(input_path), "--queries", "4", "--k", "3", "--output", str(output_path)])
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["exact_staged_search_matches_brute_force"])

    def test_evaluate_writes_recall_and_storage_report(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "vectors.npy"
            output_path = Path(directory) / "evaluation.json"
            np.save(input_path, np.random.default_rng(8).normal(size=(40, 6)))

            result = main([
                "evaluate", str(input_path), "--budgets", "16", "32",
                "--queries", "8", "--k", "3", "--repeats", "2", "--output", str(output_path),
            ])
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(report["repeats"], 2)
        self.assertEqual(len(report["results"]), 2)
        self.assertEqual(report["results"][0]["exact_rerank_recall_at_k_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()