"""Train every checkpoint required by the final ensemble in one command.

This script reproduces the final *method*. Exact accuracy can vary across complete
retraining runs because CUDA kernels, early stopping, and stochastic optimization
change the learned decision boundaries. Reusing the published local checkpoints
remains the deterministic way to reproduce the reported 24.36% evaluation.
"""

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

from models import GraphEEGNet
from src.evaluate_deep_ensemble import accuracy, load_models
from src.extract import EEGDataset
from src.train import load_and_split_data_pipeline, set_seed, train_deep_learning_model


CHECKPOINT_DIR = os.path.join(ROOT, "models", "checkpoints")


def checkpoint(name):
    return os.path.join(CHECKPOINT_DIR, name)


def train_eegnet(train_x, train_y, val_x, val_y, seed, kernel, swa, output,
                 epochs, quick):
    if os.path.exists(output):
        os.remove(output)
    set_seed(seed)
    model, _, _ = train_deep_learning_model(
        "eegnet", train_x, train_y, val_x, val_y,
        channels_count=24, time_points_count=801,
        num_epochs=epochs, batch_size=64, lr=0.005,
        temporal_kernel=kernel, use_mixup=True, mixup_alpha=0.2,
        noise_std=0.07, use_swa=swa, swa_start_epoch=25,
        quick_epochs=quick,
    )
    torch.save(model.state_dict(), output)
    print(f"Saved: {output}", flush=True)


def train_window_dcn(train_x, train_y, val_x, val_y, output, epochs, quick):
    set_seed(42)
    train_window = train_x[:, :, 50:551]
    val_window = val_x[:, :, 50:551]
    model, _, _ = train_deep_learning_model(
        "deep_conv_net", train_window, train_y, val_window, val_y,
        channels_count=24, time_points_count=501,
        num_epochs=epochs, batch_size=64, lr=0.005,
        temporal_kernel=15, use_mixup=False, noise_std=0.0,
        quick_epochs=quick,
    )
    torch.save(model.state_dict(), output)
    print(f"Saved: {output}", flush=True)


def evaluate_graph(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, 100.0 * correct / total


def train_graph(train_x, train_y, val_x, val_y, output, epochs, quick):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_window = train_x[:, :, 50:551].astype(np.float32)
    val_window = val_x[:, :, 50:551].astype(np.float32)
    train_loader = DataLoader(EEGDataset(train_window, train_y), 64, shuffle=True)
    val_loader = DataLoader(EEGDataset(val_window, val_y), 128, shuffle=False)
    model = GraphEEGNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-5
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_loss, stale = float("inf"), 0
    run_epochs = quick if quick > 0 else epochs
    temporary = output + ".training.pth"
    for epoch in range(1, run_epochs + 1):
        start = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        val_loss, val_acc = evaluate_graph(model, val_loader, device)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save(model.state_dict(), temporary)
        else:
            stale += 1
        print(
            f"GraphEEGNet epoch={epoch:03d} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.2f}% time={time.time()-start:.1f}s",
            flush=True,
        )
        if stale >= 20:
            break
    model.load_state_dict(torch.load(temporary, map_location=device))
    torch.save(model.state_dict(), output)
    os.remove(temporary)
    print(f"Saved: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train all five models used by the final EEG ensemble."
    )
    parser.add_argument("--reuse", action="store_true",
                        help="Skip checkpoints that already exist")
    parser.add_argument("--quick", type=int, default=0,
                        help="Use N epochs per model for a pipeline smoke test")
    parser.add_argument("--eegnet-epochs", type=int, default=150)
    parser.add_argument("--dcn-epochs", type=int, default=110)
    parser.add_argument("--graph-epochs", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    data_path = os.path.join(ROOT, "data", "processed", "eeg_dataset.npz")
    train_x, train_y, val_x, val_y, test_x, test_y, _, _ = (
        load_and_split_data_pipeline(data_path)
    )

    jobs = [
        ("EEGNet k=25 seed=42", checkpoint("best_eegnet_k25_seed42.pth"),
         lambda path: train_eegnet(train_x, train_y, val_x, val_y, 42, 25,
                                   False, path, args.eegnet_epochs, args.quick)),
        ("EEGNet k=15 SWA seed=42",
         checkpoint("eegnet_k15_swa_seed42_standalone.pth"),
         lambda path: train_eegnet(train_x, train_y, val_x, val_y, 42, 15,
                                   True, path, args.eegnet_epochs, args.quick)),
        ("EEGNet k=15 SWA seed=123", checkpoint("eegnet_k15_swa_seed123.pth"),
         lambda path: train_eegnet(train_x, train_y, val_x, val_y, 123, 15,
                                   True, path, args.eegnet_epochs, args.quick)),
        ("DeepConvNet 0--2000 ms", checkpoint("dcn_window_0_2000_seed42.pth"),
         lambda path: train_window_dcn(train_x, train_y, val_x, val_y, path,
                                       args.dcn_epochs, args.quick)),
        ("GraphEEGNet 0--2000 ms", checkpoint("graph_eeg_seed42.pth"),
         lambda path: train_graph(train_x, train_y, val_x, val_y, path,
                                  args.graph_epochs, args.quick)),
    ]

    for index, (name, path, trainer) in enumerate(jobs, 1):
        print(f"\n{'=' * 72}\n[{index}/5] {name}\n{'=' * 72}", flush=True)
        if args.reuse and os.path.exists(path):
            print(f"Reusing: {path}", flush=True)
        else:
            trainer(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)
    print(f"\n{'=' * 72}\nFINAL FIXED ENSEMBLE\n{'=' * 72}")
    print("Models:", ", ".join(f"{name}={weight}" for name, weight, _, _ in models))
    print(f"Validation accuracy: {accuracy(models, val_x, val_y, device):.2f}%")
    print(f"Test accuracy: {accuracy(models, test_x, test_y, device):.2f}%")


if __name__ == "__main__":
    main()
