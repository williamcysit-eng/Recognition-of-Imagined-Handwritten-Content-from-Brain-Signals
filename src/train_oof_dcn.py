import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models import DeepConvNet
from src.extract import EEGDataset
from src.train import load_and_split_data_pipeline, set_seed, train_deep_learning_model


def make_oof_splits(labels, folds=5):
    """Keep the last 10% per class as test; split the first 90% chronologically."""
    fold_val = [[] for _ in range(folds)]
    test_idx = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        development, test = indices[:270], indices[270:]
        for fold, part in enumerate(np.array_split(development, folds)):
            fold_val[fold].extend(part.tolist())
        test_idx.extend(test.tolist())
    development_idx = np.concatenate([np.asarray(x) for x in fold_val])
    splits = []
    for fold in range(folds):
        val = np.asarray(fold_val[fold])
        train = np.setdiff1d(development_idx, val, assume_unique=False)
        splits.append((train, val))
    return splits, np.asarray(test_idx)


def predict(model, x, y, device):
    logits, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in DataLoader(EEGDataset(x, y), batch_size=128):
            logits.append(model(batch_x.to(device)).cpu())
            targets.append(batch_y)
    return torch.cat(logits), torch.cat(targets)


def load_fold(path, device):
    model = DeepConvNet(24, 26, 501, temporal_kernel=15, dropout_rate=0.5)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    archive = np.load(os.path.join(ROOT, "data", "processed", "eeg_dataset.npz"))
    data = archive["data"].astype(np.float32)[:, :, 50:551]  # 0..2000 ms
    labels = archive["labels_0indexed"]
    splits, test_idx = make_oof_splits(labels, args.folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join(ROOT, "models", "checkpoints", "oof_dcn_0_2000")
    os.makedirs(checkpoint_dir, exist_ok=True)

    oof_correct = oof_total = 0
    test_logits = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        checkpoint = os.path.join(checkpoint_dir, f"fold_{fold}_seed_{args.seed + fold}.pth")
        print(f"\n{'=' * 24} FOLD {fold + 1}/{args.folds} {'=' * 24}", flush=True)
        print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
        if args.reuse and os.path.exists(checkpoint):
            model = load_fold(checkpoint, device)
        else:
            set_seed(args.seed + fold)
            model, _, device = train_deep_learning_model(
                "deep_conv_net", data[train_idx], labels[train_idx],
                data[val_idx], labels[val_idx], 24, 501,
                num_epochs=args.epochs, batch_size=64, lr=0.005,
                temporal_kernel=15, use_mixup=False, noise_std=0.0,
                early_stopping_patience=20,
            )
            torch.save(model.state_dict(), checkpoint)
        val_logits, val_targets = predict(model, data[val_idx], labels[val_idx], device)
        fold_correct = (val_logits.argmax(1) == val_targets).sum().item()
        oof_correct += fold_correct
        oof_total += len(val_targets)
        print(f"fold_accuracy={100 * fold_correct / len(val_targets):.2f}%", flush=True)
        logits, _ = predict(model, data[test_idx], labels[test_idx], device)
        test_logits.append(logits)

    test_targets = torch.from_numpy(labels[test_idx]).long()
    averaged = torch.stack(test_logits).mean(0)
    test_accuracy = 100 * (averaged.argmax(1) == test_targets).float().mean().item()
    print(f"\nOOF accuracy: {100 * oof_correct / oof_total:.2f}% ({oof_correct}/{oof_total})")
    print(f"Fold-ensemble test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
