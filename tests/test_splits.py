import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.train import load_and_split_data_pipeline
from src.train_oof_dcn import make_oof_splits


class SplitTests(unittest.TestCase):
    def test_primary_split_is_balanced_and_class_ordered(self):
        labels = np.repeat(np.arange(26, dtype=np.uint8), 10)
        data = np.zeros((len(labels), 24, 9), dtype=np.float32)
        channels = np.asarray([f"C{i}" for i in range(24)])
        times = np.arange(9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            np.savez(
                path,
                data=data,
                labels_0indexed=labels,
                labels_1indexed=labels + 1,
                channels=channels,
                time_points=times,
            )
            train_x, train_y, val_x, val_y, test_x, test_y, _, _ = (
                load_and_split_data_pipeline(path)
            )
        self.assertEqual(train_x.shape, (208, 24, 9))
        self.assertEqual(val_x.shape, (26, 24, 9))
        self.assertEqual(test_x.shape, (26, 24, 9))
        np.testing.assert_array_equal(np.bincount(train_y), np.full(26, 8))
        np.testing.assert_array_equal(np.bincount(val_y), np.ones(26, dtype=int))
        np.testing.assert_array_equal(np.bincount(test_y), np.ones(26, dtype=int))

    def test_five_fold_oof_has_no_overlap_and_keeps_last_30_for_test(self):
        labels = np.repeat(np.arange(26, dtype=np.uint8), 300)
        splits, test_idx = make_oof_splits(labels, folds=5)
        self.assertEqual(len(test_idx), 780)
        validation = []
        for train_idx, val_idx in splits:
            self.assertEqual(len(train_idx), 5616)
            self.assertEqual(len(val_idx), 1404)
            self.assertEqual(len(np.intersect1d(train_idx, val_idx)), 0)
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
            self.assertEqual(len(np.intersect1d(val_idx, test_idx)), 0)
            validation.append(val_idx)
        development = np.concatenate(validation)
        self.assertEqual(len(np.unique(development)), 7020)
        self.assertEqual(len(np.intersect1d(development, test_idx)), 0)
        for label in range(26):
            indices = np.flatnonzero(labels == label)
            np.testing.assert_array_equal(test_idx[labels[test_idx] == label], indices[270:])


if __name__ == "__main__":
    unittest.main()
