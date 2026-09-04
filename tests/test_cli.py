import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_train_help_renders(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "train.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--temporal-kernel", result.stdout)
        self.assertIn("--run-id", result.stdout)

    def test_best_hybrid_help_renders(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "train_best_hybrid.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--force", result.stdout)
        self.assertIn("--dry-run", result.stdout)


if __name__ == "__main__":
    unittest.main()
