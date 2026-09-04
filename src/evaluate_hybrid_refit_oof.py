"""Evaluate the fixed 50/50 OOF and full-development-refit hybrid."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models import DeepConvNet, EEGNet82, GraphEEGNet
from src.checkpoint_manifest import (
    BEST_HYBRID_ARCHITECTURES,
    BEST_HYBRID_SCALES,
    BEST_HYBRID_WEIGHTS,
)
from src.extract import EEGDataset
from src.train_oof_aligned_dcn import align


def predict(model, x, y, device):
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch_x, _ in DataLoader(EEGDataset(x, y), 128):
            outputs.append(model(batch_x.to(device)).cpu())
    return torch.cat(outputs)


def _load(model, path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required full-refit checkpoint is missing: {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device)


def evaluate_hybrid(cache_path, checkpoint_root, data_path, *, force_cpu=False):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"OOF cache not found: {cache_path}. Run evaluate_oof_multiarch_ensemble.py first."
        )
    cache = np.load(cache_path, allow_pickle=False)
    names = tuple(str(value) for value in cache["names"]) if "names" in cache else BEST_HYBRID_ARCHITECTURES
    if names != tuple(BEST_HYBRID_ARCHITECTURES):
        raise ValueError(f"Unexpected OOF architecture order: {names}")
    weights = cache["weights"].astype(np.float32) if "weights" in cache else np.asarray(BEST_HYBRID_WEIGHTS, dtype=np.float32)
    scales = cache["scales"].astype(np.float32) if "scales" in cache else np.asarray(BEST_HYBRID_SCALES, dtype=np.float32)
    oof_raw = torch.tensor(cache["test"])
    target = torch.tensor(cache["test_targets"])
    oof = (
        oof_raw
        * torch.tensor(scales)[None, :, None]
        * torch.tensor(weights)[None, :, None]
    ).sum(1)

    archive = np.load(data_path, allow_pickle=False)
    data = archive["data"].astype(np.float32)
    labels = archive["labels_0indexed"]
    test = np.concatenate([np.flatnonzero(labels == label)[270:] for label in range(26)])
    raw = data[test, :, 50:551]
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    refit_root = Path(checkpoint_root) / "full_refit"

    full = torch.zeros_like(oof)
    if weights[0] > 0:
        dcn_predictions = []
        for seed in (42, 542):
            model = _load(
                DeepConvNet(24, 26, 501, temporal_kernel=15, dropout_rate=0.5),
                refit_root / f"dcn_seed{seed}.pth",
                device,
            )
            dcn_predictions.append(predict(model, raw, labels[test], device))
        full += weights[0] * scales[0] * torch.stack(dcn_predictions).mean(0)
    if weights[1] > 0:
        model = _load(
            DeepConvNet(24, 26, 501, temporal_kernel=15, dropout_rate=0.5),
            refit_root / "aligned_dcn_seed442.pth",
            device,
        )
        full += weights[1] * scales[1] * predict(model, align(raw), labels[test], device)
    if weights[2] > 0:
        model = _load(
            EEGNet82(24, 26, input_time_points=801, temporal_kernel_length=25, dropout_rate=0.3),
            refit_root / "k25_seed142.pth",
            device,
        )
        full += weights[2] * scales[2] * predict(model, data[test], labels[test], device)
    if weights[3] > 0:
        k15_predictions = []
        for seed in (342, 642):
            model = _load(
                EEGNet82(24, 26, input_time_points=801, temporal_kernel_length=15, dropout_rate=0.3),
                refit_root / f"k15_swa_seed{seed}.pth",
                device,
            )
            k15_predictions.append(predict(model, data[test], labels[test], device))
        full += weights[3] * scales[3] * torch.stack(k15_predictions).mean(0)
    if weights[4] > 0:
        model = _load(GraphEEGNet(), refit_root / "graph_seed242.pth", device)
        full += weights[4] * scales[4] * predict(model, raw, labels[test], device)

    full = full * (oof.std() / full.std())
    hybrid = (oof + full) / 2
    result = {
        "weights": tuple(int(value) for value in weights),
        "scales": tuple(float(value) for value in scales),
        "oof_test_accuracy": 100 * (oof.argmax(1) == target).float().mean().item(),
        "full_refit_test_accuracy": 100 * (full.argmax(1) == target).float().mean().item(),
        "hybrid_test_accuracy": 100 * (hybrid.argmax(1) == target).float().mean().item(),
    }
    print("Weights:", result["weights"])
    print(f"OOF fold ensemble: {result['oof_test_accuracy']:.2f}%")
    print(f"Full-development refit: {result['full_refit_test_accuracy']:.2f}%")
    print(f"Fixed 50/50 hybrid: {result['hybrid_test_accuracy']:.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", default=ROOT / "outputs" / "oof_multiarch_logits.npz")
    parser.add_argument("--checkpoint-root", default=ROOT / "models" / "checkpoints")
    parser.add_argument("--data-path", default=ROOT / "data" / "processed" / "eeg_dataset.npz")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    evaluate_hybrid(
        args.cache_path,
        args.checkpoint_root,
        args.data_path,
        force_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
