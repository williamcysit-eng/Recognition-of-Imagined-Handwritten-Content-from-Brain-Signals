import tempfile
import unittest
from pathlib import Path

from src.run_utils import guard_output, prepare_run_dir


class RunOutputTests(unittest.TestCase):
    def test_existing_run_is_rejected_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            run = prepare_run_dir("test", "fixed", directory)
            (run / "artifact.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_run_dir("test", "fixed", directory)
            self.assertEqual((run / "artifact.txt").read_text(encoding="utf-8"), "keep")
            self.assertEqual(prepare_run_dir("test", "fixed", directory, force=True), run)

    def test_existing_output_is_rejected_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.pth"
            output.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                guard_output(output)
            self.assertEqual(output.read_bytes(), b"original")
            self.assertTrue(guard_output(output, force=True))


if __name__ == "__main__":
    unittest.main()
