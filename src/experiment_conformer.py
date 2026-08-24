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

from models.eeg_conformer import EEGConformer
from src.extract import EEGDataset
from src.train import load_and_split_data_pipeline, set_seed


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, 100 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(ROOT, "data", "processed", "eeg_dataset.npz")
    train_x, train_y, val_x, val_y, test_x, test_y, _, _ = load_and_split_data_pipeline(path)
    train_x, val_x, test_x = (x[:, :, 50:551].astype(np.float32)
                              for x in (train_x, val_x, test_x))
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True).clip(1e-6)
    train_x, val_x, test_x = ((x - mean) / std for x in (train_x, val_x, test_x))
    train_loader = DataLoader(EEGDataset(train_x, train_y), 64, shuffle=True)
    val_loader = DataLoader(EEGDataset(val_x, val_y), 128)
    test_loader = DataLoader(EEGDataset(test_x, test_y), 128)

    model = EEGConformer(dropout=0.15).to(device)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 1e-5)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint = os.path.join(ROOT, "models", "checkpoints",
                              f"eeg_conformer_0_2000_seed{args.seed}.pth")
    best_loss, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        print(f"epoch={epoch:03d} val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
              f"time={time.time()-start:.1f}s", flush=True)
        if stale >= 20:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print("BEST validation:", evaluate(model, val_loader, device))
    print("checkpoint:", checkpoint)
    if args.test:
        print("FINAL test:", evaluate(model, test_loader, device))


if __name__ == "__main__":
    main()
