"""Train the historical fixed five-model, single-split ensemble.

The current best hybrid is trained by ``train_best_hybrid.py``. This script remains
for the retained 24.36% experiment. Exact accuracy can vary across complete
retraining runs because CUDA kernels, early stopping, and stochastic optimization
change the learned decision boundaries. Outputs are isolated under ``runs/``.
"""

import argparse
import copy
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
from src.run_utils import base_manifest, guard_output, prepare_run_dir, save_torch_state, write_json
from src.train import load_and_split_data_pipeline, set_seed, train_deep_learning_model


CHECKPOINT_DIR = None


def checkpoint(name):
    return os.path.join(CHECKPOINT_DIR, name)


def train_eegnet(train_x, train_y, val_x, val_y, seed, kernel, swa, output,
                 epochs, quick, force=False):
    set_seed(seed)
    model, _, _ = train_deep_learning_model(
        "eegnet", train_x, train_y, val_x, val_y,
        channels_count=24, time_points_count=801,
        num_epochs=epochs, batch_size=64, lr=0.005,
        temporal_kernel=kernel, use_mixup=True, mixup_alpha=0.2,
        noise_std=0.07, use_swa=swa, swa_start_epoch=25,
        quick_epochs=quick,
    )
    save_torch_state(model, output, force=force)
    print(f"Saved: {output}", flush=True)


def train_window_dcn(train_x, train_y, val_x, val_y, output, epochs, quick, force=False):
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
    save_torch_state(model, output, force=force)
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


def train_graph(train_x, train_y, val_x, val_y, output, epochs, quick, force=False):
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
    best_state = None
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
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(
            f"GraphEEGNet epoch={epoch:03d} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.2f}% time={time.time()-start:.1f}s",
            flush=True,
        )
        if stale >= 20:
            break
    if best_state is None:
        raise RuntimeError("Graph training produced no validation checkpoint")
    model.load_state_dict(best_state)
    save_torch_state(model, output, force=force)
    print(f"Saved: {output}", flush=True)


def main():
    global CHECKPOINT_DIR
    parser = argparse.ArgumentParser(
        description="Train the historical fixed five-model EEG ensemble."
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--reuse", action="store_true",
                        help="Reuse checkpoints in an explicitly selected run")
    output.add_argument("--force", action="store_true",
                        help="Allow overwriting checkpoints in an existing run")
    parser.add_argument("--quick", type=int, default=0,
                        help="Use N epochs per model for a pipeline smoke test")
    parser.add_argument("--eegnet-epochs", type=int, default=150)
    parser.add_argument("--dcn-epochs", type=int, default=110)
    parser.add_argument("--graph-epochs", type=int, default=100)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-root", default=os.path.join(ROOT, "runs"))
    args = parser.parse_args()
    run_dir = prepare_run_dir(
        "legacy-five-model", args.run_id, args.runs_root, args.force or args.reuse
    )
    CHECKPOINT_DIR = os.path.join(run_dir, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Run directory: {run_dir}")

    data_path = os.path.join(ROOT, "data", "processed", "eeg_dataset.npz")
    train_x, train_y, val_x, val_y, test_x, test_y, _, _ = (
        load_and_split_data_pipeline(data_path)
    )

    jobs = [
        ("EEGNet k=25 seed=42", checkpoint("best_eegnet_k25_seed42.pth"),
         lambda path: train_eegnet(train_x, train_y, val_x, val_y, 42, 25,
                                   False, path, args.eegnet_epochs, args.quick, args.force)),
        ("EEGNet k=15 SWA seed=42",
         checkpoint("eegnet_k15_swa_seed42_standalone.pth"),
         lambda path: train_eegnet(train_x, train_y, val_x, val_y, 42, 15,
                                   True, path, args.eegnet_epochs, args.quick, args.force)),
        ("EEGNet k=15 SWA seed=123", checkpoint("eegnet_k15_swa_seed123.pth"),
         lambda path: train_eegnet(train_x, train_y, val_x, val_y, 123, 15,
                                   True, path, args.eegnet_epochs, args.quick, args.force)),
        ("DeepConvNet 0--2000 ms", checkpoint("dcn_window_0_2000_seed42.pth"),
         lambda path: train_window_dcn(train_x, train_y, val_x, val_y, path,
                                       args.dcn_epochs, args.quick, args.force)),
        ("GraphEEGNet 0--2000 ms", checkpoint("graph_eeg_seed42.pth"),
         lambda path: train_graph(train_x, train_y, val_x, val_y, path,
                                  args.graph_epochs, args.quick, args.force)),
    ]

    for index, (name, path, trainer) in enumerate(jobs, 1):
        print(f"\n{'=' * 72}\n[{index}/5] {name}\n{'=' * 72}", flush=True)
        if args.reuse and os.path.exists(path):
            print(f"Reusing: {path}", flush=True)
        else:
            guard_output(path, force=args.force)
            trainer(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device, CHECKPOINT_DIR)
    print(f"\n{'=' * 72}\nFINAL FIXED ENSEMBLE\n{'=' * 72}")
    print("Models:", ", ".join(f"{name}={weight}" for name, weight, _, _ in models))
    validation_accuracy = accuracy(models, val_x, val_y, device)
    test_accuracy = accuracy(models, test_x, test_y, device)
    print(f"Validation accuracy: {validation_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    metrics = {"validation_accuracy": validation_accuracy, "test_accuracy": test_accuracy}
    manifest = base_manifest(run_dir.name, sys.argv, vars(args));manifest["results"] = metrics
    manifest_path = os.path.join(run_dir, "manifest.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not args.reuse or not os.path.exists(manifest_path):
        write_json(manifest_path, manifest, force=args.force)
    if not args.reuse or not os.path.exists(metrics_path):
        write_json(metrics_path, metrics, force=args.force)


if __name__ == "__main__":
    main()
