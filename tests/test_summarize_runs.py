import json
import tempfile
import unittest
from pathlib import Path

from src.summarize_runs import summarize


class SummarizeRunsTests(unittest.TestCase):
    def test_reports_mean_sample_sd_interval_and_individual_values(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, value in enumerate((28.0, 29.0, 30.0)):
                path = Path(directory) / f"run-{index}.json"
                path.write_text(
                    json.dumps({"hybrid": {"hybrid_test_accuracy": value}}),
                    encoding="utf-8",
                )
                paths.append(path)
            result = summarize(paths, "hybrid.hybrid_test_accuracy")
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 29.0)
        self.assertEqual(result["sample_sd"], 1.0)
        self.assertEqual([run["value"] for run in result["runs"]], [28.0, 29.0, 30.0])
        self.assertLess(result["confidence_interval_95"][0], 29.0)
        self.assertGreater(result["confidence_interval_95"][1], 29.0)


if __name__ == "__main__":
    unittest.main()
