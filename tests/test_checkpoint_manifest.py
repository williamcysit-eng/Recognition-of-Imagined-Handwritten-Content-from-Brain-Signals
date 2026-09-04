import unittest
from pathlib import Path

from src.checkpoint_manifest import (
    FULL_REFIT_CHECKPOINTS,
    LEGACY_FIVE_MODEL_CHECKPOINTS,
    expected_oof_checkpoint_paths,
    full_refit_checkpoint_names,
)


ROOT = Path(__file__).resolve().parents[1]


class CheckpointManifestTests(unittest.TestCase):
    def test_manifest_counts_and_names_are_unique(self):
        oof = expected_oof_checkpoint_paths()
        self.assertEqual(len(oof), 35)
        self.assertEqual(len(set(oof)), 35)
        self.assertEqual(len(FULL_REFIT_CHECKPOINTS), 6)
        self.assertEqual(full_refit_checkpoint_names(), FULL_REFIT_CHECKPOINTS)
        self.assertEqual(
            full_refit_checkpoint_names((0, 2, 2, 4, 12)),
            (
                "aligned_dcn_seed442.pth",
                "k25_seed142.pth",
                "k15_swa_seed342.pth",
                "k15_swa_seed642.pth",
                "graph_seed242.pth",
            ),
        )
        legacy_names = [entry[0] for entry in LEGACY_FIVE_MODEL_CHECKPOINTS]
        legacy_files = [entry[2] for entry in LEGACY_FIVE_MODEL_CHECKPOINTS]
        self.assertEqual(len(legacy_names), len(set(legacy_names)))
        self.assertEqual(len(legacy_files), len(set(legacy_files)))

    def test_local_published_checkpoint_set_is_complete_when_present(self):
        checkpoint_root = ROOT / "models" / "checkpoints"
        expected = [checkpoint_root / path for path in expected_oof_checkpoint_paths()]
        expected += [checkpoint_root / "full_refit" / name for name in FULL_REFIT_CHECKPOINTS]
        if not any(path.exists() for path in expected):
            self.skipTest("Published local checkpoints are not installed")
        missing = [str(path) for path in expected if not path.exists()]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
