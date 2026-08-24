import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.eeg_res_tcn import EEGResTCN
from models.shallow_conv_net import ShallowConvNet
from models.filter_bank_net import FilterBankNet
from src.extract import EEGDataset
from src.train import load_and_split_data_pipeline, set_seed


def normalize(train, val, test, mode):
    if mode == "none":
        return train, val, test
    if mode == "train_channel":
        mean = train.mean(axis=(0, 2), keepdims=True)
        std = train.std(axis=(0, 2), keepdims=True).clip(1e-6)
        return tuple(((x - mean) / std).astype(np.float32) for x in (train, val, test))
    if mode == "trial_channel":
        def transform(x):
            mean = x.mean(axis=2, keepdims=True)
            std = x.std(axis=2, keepdims=True).clip(1e-6)
            return ((x - mean) / std).astype(np.float32)
        return tuple(transform(x) for x in (train, val, test))
    raise ValueError(mode)


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, 100 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--model", choices=["res_tcn", "shallow", "filter_bank"],
                        default="res_tcn")
    parser.add_argument("--normalization", choices=["none", "train_channel", "trial_channel"],
                        default="train_channel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true",
                        help="Evaluate the selected best checkpoint on the test split")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = os.path.join(ROOT, "data", "processed", "eeg_dataset.npz")
    train_x, train_y, val_x, val_y, test_x, test_y, _, _ = load_and_split_data_pipeline(path)
    train_x, val_x, test_x = normalize(train_x, val_x, test_x, args.normalization)
    train_loader = DataLoader(EEGDataset(train_x, train_y), batch_size=64, shuffle=True)
    val_loader = DataLoader(EEGDataset(val_x, val_y), batch_size=128)
    test_loader = DataLoader(EEGDataset(test_x, test_y), batch_size=128)

    if args.model == "res_tcn":
        model = EEGResTCN()
    elif args.model == "shallow":
        model = ShallowConvNet()
    else:
        model = FilterBankNet()
    model = model.to(device)
    print(f"device={device} normalization={args.normalization} parameters={sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    checkpoint = os.path.join(ROOT, "models", "checkpoints", f"best_{args.model}_{args.normalization}.pth")
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint)
            stale = 0
        else:
            stale += 1
        print(f"epoch={epoch:03d} val_loss={val_loss:.4f} val_acc={val_acc:.2f}% time={time.time()-start:.1f}s")
        if stale >= 15:
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"BEST validation loss={val_loss:.4f} accuracy={val_acc:.2f}% checkpoint={checkpoint}")
    if args.test:
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"FINAL test loss={test_loss:.4f} accuracy={test_acc:.2f}%")


if __name__ == "__main__":
    main()
