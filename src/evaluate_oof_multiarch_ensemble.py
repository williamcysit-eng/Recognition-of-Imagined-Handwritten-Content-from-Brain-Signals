"""Select and evaluate the leakage-free multi-architecture OOF ensemble."""

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
from src.checkpoint_manifest import BEST_HYBRID_ARCHITECTURES
from src.extract import EEGDataset
from src.run_utils import guard_output, prepare_run_dir
from src.train_oof_aligned_dcn import align
from src.train_oof_dcn import make_oof_splits


NAMES = BEST_HYBRID_ARCHITECTURES
DCN_SEEDS = (42, 542)
K15_SEEDS = (342, 642)


def predict(model, data, labels, device):
    outputs = []
    model.eval()
    with torch.no_grad():
        loader = DataLoader(EEGDataset(data, labels), batch_size=128)
        for batch_x, _ in loader:
            outputs.append(model(batch_x.to(device)).cpu())
    return torch.cat(outputs)


def seed_average(models, data, labels, device):
    return torch.stack([predict(model, data, labels, device) for model in models]).mean(0)


def compositions(total, count, prefix=()):
    if count == 1:
        yield prefix + (total,)
        return
    for value in range(total + 1):
        yield from compositions(total - value, count - 1, prefix + (value,))


def _load(model, path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required checkpoint is missing: {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device)


def _atomic_savez(path, force=False, **arrays):
    output = Path(path)
    guard_output(output, force=force)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, output)


def evaluate_oof(
    checkpoint_root,
    data_path,
    cache_path=None,
    *,
    force=False,
    force_cpu=False,
    folds=5,
    weight_budget=20,
):
    checkpoint_root = Path(checkpoint_root)
    archive = np.load(data_path, allow_pickle=False)
    data = archive["data"].astype(np.float32)
    labels = archive["labels_0indexed"]
    splits, test_idx = make_oof_splits(labels, folds)
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")

    oof = [[] for _ in NAMES]
    tests = [[] for _ in NAMES]
    targets = []
    fold_ids = []

    for fold, (_, val_idx) in enumerate(splits):
        dcns = [
            _load(
                DeepConvNet(24, 26, 501, temporal_kernel=15, dropout_rate=0.5),
                checkpoint_root / "oof_dcn_0_2000" / f"fold_{fold}_seed_{seed + fold}.pth",
                device,
            )
            for seed in DCN_SEEDS
        ]
        aligned = _load(
            DeepConvNet(24, 26, 501, temporal_kernel=15, dropout_rate=0.5),
            checkpoint_root / "oof_aligned_dcn" / f"fold_{fold}_seed_{442 + fold}.pth",
            device,
        )
        k25 = _load(
            EEGNet82(24, 26, input_time_points=801, temporal_kernel_length=25, dropout_rate=0.3),
            checkpoint_root / "oof_eegnet_k25" / f"fold_{fold}_seed_{142 + fold}.pth",
            device,
        )
        k15s = [
            _load(
                EEGNet82(24, 26, input_time_points=801, temporal_kernel_length=15, dropout_rate=0.3),
                checkpoint_root / "oof_eegnet_k15_swa" / f"fold_{fold}_seed_{seed + fold}.pth",
                device,
            )
            for seed in K15_SEEDS
        ]
        graph = _load(
            GraphEEGNet(),
            checkpoint_root / "oof_graph" / f"fold_{fold}_seed_{242 + fold}.pth",
            device,
        )

        raw_val = data[val_idx, :, 50:551]
        raw_test = data[test_idx, :, 50:551]
        targets.append(torch.from_numpy(labels[val_idx]).long())
        fold_ids.append(np.full(len(val_idx), fold, dtype=np.int16))

        oof[0].append(seed_average(dcns, raw_val, labels[val_idx], device))
        tests[0].append(seed_average(dcns, raw_test, labels[test_idx], device))
        inputs = ((aligned, align(raw_val), align(raw_test)), (k25, data[val_idx], data[test_idx]))
        for index, (model, val_x, test_x) in enumerate(inputs, start=1):
            oof[index].append(predict(model, val_x, labels[val_idx], device))
            tests[index].append(predict(model, test_x, labels[test_idx], device))
        oof[3].append(seed_average(k15s, data[val_idx], labels[val_idx], device))
        tests[3].append(seed_average(k15s, data[test_idx], labels[test_idx], device))
        oof[4].append(predict(graph, raw_val, labels[val_idx], device))
        tests[4].append(predict(graph, raw_test, labels[test_idx], device))
        print(f"Loaded fold {fold + 1}/{folds}", flush=True)

    oof = [torch.cat(item) for item in oof]
    target = torch.cat(targets)
    raw_test = [torch.stack(item).mean(0) for item in tests]
    scales = [oof[0].std() / item.std() for item in oof]
    scaled_oof = [item * scale for item, scale in zip(oof, scales)]

    best_accuracy, best_weights = -1.0, None
    for weights in compositions(weight_budget, len(NAMES)):
        logits = sum(weight * item for weight, item in zip(weights, scaled_oof))
        accuracy = 100 * (logits.argmax(1) == target).float().mean().item()
        if accuracy > best_accuracy:
            best_accuracy, best_weights = accuracy, weights

    scaled_test = [item * scale for item, scale in zip(raw_test, scales)]
    test_target = torch.from_numpy(labels[test_idx]).long()
    test_logits = sum(weight * item for weight, item in zip(best_weights, scaled_test))
    test_accuracy = 100 * (test_logits.argmax(1) == test_target).float().mean().item()

    if cache_path is not None:
        _atomic_savez(
            cache_path,
            force=force,
            oof=np.stack([item.numpy() for item in oof], axis=1),
            test=np.stack([item.numpy() for item in raw_test], axis=1),
            oof_targets=target.numpy(),
            test_targets=labels[test_idx],
            fold_ids=np.concatenate(fold_ids),
            names=np.asarray(NAMES),
            scales=np.asarray([float(value) for value in scales], dtype=np.float32),
            weights=np.asarray(best_weights, dtype=np.int16),
        )

    result = {
        "names": tuple(NAMES),
        "scales": tuple(float(value) for value in scales),
        "weights": tuple(int(value) for value in best_weights),
        "oof_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
    }
    print("Architectures:", ", ".join(result["names"]))
    print("OOF scales:", ", ".join(f"{value:.6f}" for value in result["scales"]))
    print("OOF-selected integer weights:", result["weights"])
    print(f"OOF accuracy: {best_accuracy:.2f}%")
    print(f"Held-out test accuracy: {test_accuracy:.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default=ROOT / "models" / "checkpoints")
    parser.add_argument("--data-path", default=ROOT / "data" / "processed" / "eeg_dataset.npz")
    parser.add_argument("--cache-path", help="Explicit output .npz path")
    parser.add_argument("--run-id")
    parser.add_argument("--runs-root", default=ROOT / "runs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.cache_path:
        cache_path = Path(args.cache_path)
    else:
        run_dir = prepare_run_dir("oof-eval", args.run_id, args.runs_root, args.force)
        cache_path = run_dir / "outputs" / "oof_multiarch_logits.npz"
        print(f"Run directory: {run_dir}")
    evaluate_oof(
        args.checkpoint_root,
        args.data_path,
        cache_path,
        force=args.force,
        force_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
