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

from models.graph_eeg_net import GraphEEGNet, DynamicGraphEEGNet, MultiScaleGraphEEGNet
from src.extract import EEGDataset
from src.run_utils import guard_output, prepare_run_dir
from src.train import load_and_split_data_pipeline, set_seed


def evaluate(model, loader, device):
    model.eval(); correct = total = 0; loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device); logits = model(x)
            loss_sum += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item(); total += len(y)
    return loss_sum / total, 100 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--run-id");parser.add_argument("--runs-root",default=os.path.join(ROOT,"runs"));parser.add_argument("--force",action="store_true")
    args = parser.parse_args(); set_seed(args.seed)
    run_dir=prepare_run_dir('experiment-graph',args.run_id,args.runs_root,args.force);print(f'Run directory: {run_dir}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(ROOT, "data", "processed", "eeg_dataset.npz")
    tr, ty, va, vy, te, tey, _, _ = load_and_split_data_pipeline(path)
    tr, va, te = (x[:, :, 50:551].astype(np.float32) for x in (tr, va, te))
    model = (MultiScaleGraphEEGNet() if args.multiscale else DynamicGraphEEGNet() if args.dynamic else GraphEEGNet()).to(device)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    train_loader = DataLoader(EEGDataset(tr, ty), 64, shuffle=True)
    val_loader = DataLoader(EEGDataset(va, vy), 128)
    test_loader = DataLoader(EEGDataset(te, tey), 128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                            patience=3, min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    prefix = "multiscale_graph_eeg" if args.multiscale else "dynamic_graph_eeg" if args.dynamic else "graph_eeg"
    checkpoint = os.path.join(run_dir, "checkpoints", f"{prefix}_seed{args.seed}.pth")
    guard_output(checkpoint,force=args.force)
    best_loss, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        start = time.time(); model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        val_loss, val_acc = evaluate(model, val_loader, device); scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0; torch.save(model.state_dict(), checkpoint)
        else: stale += 1
        print(f"epoch={epoch:03d} val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
              f"time={time.time()-start:.1f}s", flush=True)
        if stale >= 20: break
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print("BEST validation:", evaluate(model, val_loader, device)); print("checkpoint:", checkpoint)
    if args.test: print("FINAL test:", evaluate(model, test_loader, device))


if __name__ == "__main__": main()
