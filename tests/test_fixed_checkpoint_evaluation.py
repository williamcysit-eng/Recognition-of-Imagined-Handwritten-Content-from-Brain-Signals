import unittest
from pathlib import Path

import torch

from src.evaluate_deep_ensemble import accuracy, load_models
from src.evaluate_hybrid_refit_oof import evaluate_hybrid
from src.train import load_and_split_data_pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "eeg_dataset.npz"
CHECKPOINTS = ROOT / "models" / "checkpoints"
CACHE = ROOT / "outputs" / "oof_multiarch_logits.npz"
ASSETS_AVAILABLE = (
    DATA.exists()
    and CACHE.exists()
    and (CHECKPOINTS / "best_eegnet_k25_seed42.pth").exists()
    and (CHECKPOINTS / "full_refit" / "k25_seed142.pth").exists()
)


@unittest.skipUnless(ASSETS_AVAILABLE, "Fixed local evaluation assets are not installed")
class FixedCheckpointEvaluationTests(unittest.TestCase):
    def test_historical_five_model_result(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, _, val_x, val_y, test_x, test_y, _, _ = load_and_split_data_pipeline(DATA)
        models = load_models(device, CHECKPOINTS)
        self.assertAlmostEqual(accuracy(models, val_x, val_y, device), 23.33, places=2)
        self.assertAlmostEqual(accuracy(models, test_x, test_y, device), 24.36, places=2)

    def test_best_hybrid_result(self):
        result = evaluate_hybrid(CACHE, CHECKPOINTS, DATA)
        self.assertAlmostEqual(result["oof_test_accuracy"], 28.33, places=2)
        self.assertAlmostEqual(result["full_refit_test_accuracy"], 28.46, places=2)
        self.assertAlmostEqual(result["hybrid_test_accuracy"], 29.10, places=2)


if __name__ == "__main__":
    unittest.main()
