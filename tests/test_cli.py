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


if __name__ == "__main__":
    unittest.main()