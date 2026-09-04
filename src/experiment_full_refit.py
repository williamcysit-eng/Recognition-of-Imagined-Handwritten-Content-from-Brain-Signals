"""Refit OOF-selected architectures on all 270 development trials per class."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models import DeepConvNet, EEGNet82, GraphEEGNet, apply_max_norm_constraints
from src.checkpoint_manifest import (
    BEST_HYBRID_SCALES,
    BEST_HYBRID_WEIGHTS,
    full_refit_checkpoint_names,
)
from src.extract import EEGDataset
from src.run_utils import guard_output, prepare_run_dir, save_torch_state
from src.train import set_seed
from src.train_oof_aligned_dcn import align


def train_model(model, x, y, epochs, seed, *, eeg=False, swa=False, force_cpu=False):
    set_seed(seed)
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    model = model.to(device)
    loader = DataLoader(EEGDataset(x, y), 64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    averaged, averaged_count = None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if eeg:
                batch_x = batch_x + torch.randn_like(batch_x) * 0.07
                mix = np.random.beta(0.2, 0.2)
                index = torch.randperm(len(batch_y), device=device)
                output = model(mix * batch_x + (1 - mix) * batch_x[index])
                loss = mix * F.cross_entropy(output, batch_y, label_smoothing=0.1)
                loss += (1 - mix) * F.cross_entropy(output, batch_y[index], label_smoothing=0.1)
            else:
                output = model(batch_x)
                loss = F.cross_entropy(output, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if eeg:
                apply_max_norm_constraints(model)
        scheduler.step()

        if swa and epoch >= 25:
            state = model.state_dict()
            if averaged is None:
                averaged = {key: value.detach().clone() for key, value in state.items()}
                averaged_count = 1
            else:
                for key in averaged:
                    if torch.is_floating_point(averaged[key]):
                        averaged[key] = (
                            averaged[key] * averaged_count + state[key].detach()
                        ) / (averaged_count + 1)
                    else:
                        averaged[key] = state[key].detach().clone()
                averaged_count += 1
        if epoch % 10 == 0 or epoch == epochs:
            print(f"seed={seed} epoch={epoch}/{epochs}", flush=True)

    if averaged is not None:
        model.load_state_dict(averaged)
    return model


def predict(model, x, y):
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_x, _ in DataLoader(EEGDataset(x, y), 128):
            outputs.append(model(batch_x.to(device)).cpu())
    return torch.cat(outputs)


def train_graph_model(x, y, epochs, seed, *, force_cpu=False):
    set_seed(seed)
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    model = GraphEEGNet().to(device)
    loader = DataLoader(EEGDataset(x, y), 64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.03)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        if epoch % 10 == 0 or epoch == epochs:
            print(f"graph seed={seed} epoch={epoch}/{epochs}", flush=True)
    return model


def run_full_refit(
    checkpoint_root,
    data_path,
    *,
    force=False,
    force_cpu=False,
    dcn_epochs=40,
    k25_epochs=40,
    k15_epochs=60,
    graph_epochs=40,
    ensemble_weights=BEST_HYBRID_WEIGHTS,
    ensemble_scales=BEST_HYBRID_SCALES,
):
    weights = tuple(ensemble_weights)
    scales = tuple(ensemble_scales)
    checkpoint_dir = Path(checkpoint_root) / "full_refit"
    checkpoint_names = full_refit_checkpoint_names(weights)
    paths = {name: checkpoint_dir / name for name in checkpoint_names}
    for path in paths.values():
        guard_output(path, force=force)

    archive = np.load(data_path, allow_pickle=False)
    data = archive["data"].astype(np.float32)
    labels = archive["labels_0indexed"]
    development, test = [], []
    for label in range(26):
        indices = np.flatnonzero(labels == label)
        development.extend(indices[:270])
        test.extend(indices[270:])
    development, test = np.asarray(development), np.asarray(test)
    raw_development = data[development, :, 50:551]
    raw_test = data[test, :, 50:551]

    dcn_predictions = []
    if weights[0] > 0:
        for seed in (42, 542):
            model = train_model(
                DeepConvNet(24, 26, 501, temporal_kernel=15, dropout_rate=0.5),
                raw_development,
                labels[development],
                dcn_epochs,
                seed,
                force_cpu=force_cpu,
            )
            save_torch_state(model, paths[f"dcn_seed{seed}.pth"], force=force)
            dcn_predictions.append(predict(model, raw_test, labels[test]))

    aligned_predictions = None
    if weights[1] > 0:
        aligned = train_model(
            DeepConvNet(24, 26, 501, temporal_kernel=15, dropout_rate=0.5),
            align(raw_development),
            labels[development],
            dcn_epochs,
            442,
            force_cpu=force_cpu,
        )
        save_torch_state(aligned, paths["aligned_dcn_seed442.pth"], force=force)
        aligned_predictions = predict(aligned, align(raw_test), labels[test])

    k25_predictions = None
    if weights[2] > 0:
        k25 = train_model(
            EEGNet82(24, 26, input_time_points=801, temporal_kernel_length=25, dropout_rate=0.3),
            data[development],
            labels[development],
            k25_epochs,
            142,
            eeg=True,
            force_cpu=force_cpu,
        )
        save_torch_state(k25, paths["k25_seed142.pth"], force=force)
        k25_predictions = predict(k25, data[test], labels[test])

    k15_predictions = []
    if weights[3] > 0:
        for seed in (342, 642):
            model = train_model(
                EEGNet82(24, 26, input_time_points=801, temporal_kernel_length=15, dropout_rate=0.3),
                data[development],
                labels[development],
                k15_epochs,
                seed,
                eeg=True,
                swa=True,
                force_cpu=force_cpu,
            )
            save_torch_state(model, paths[f"k15_swa_seed{seed}.pth"], force=force)
            k15_predictions.append(predict(model, data[test], labels[test]))

    graph_predictions = None
    if weights[4] > 0:
        graph = train_graph_model(
            raw_development,
            labels[development],
            graph_epochs,
            242,
            force_cpu=force_cpu,
        )
        save_torch_state(graph, paths["graph_seed242.pth"], force=force)
        graph_predictions = predict(graph, raw_test, labels[test])

    logits = torch.zeros((len(test), 26), dtype=torch.float32)
    if dcn_predictions:
        logits += weights[0] * scales[0] * torch.stack(dcn_predictions).mean(0)
    if aligned_predictions is not None:
        logits += weights[1] * scales[1] * aligned_predictions
    if k25_predictions is not None:
        logits += weights[2] * scales[2] * k25_predictions
    if k15_predictions:
        logits += weights[3] * scales[3] * torch.stack(k15_predictions).mean(0)
    if graph_predictions is not None:
        logits += weights[4] * scales[4] * graph_predictions
    target = torch.tensor(labels[test])
    accuracy = 100 * (logits.argmax(1) == target).float().mean().item()
    print(f"FULL_REFIT_TEST={accuracy:.2f}%")
    return {"full_refit_test_accuracy": accuracy, "checkpoint_dir": str(checkpoint_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=ROOT / "data" / "processed" / "eeg_dataset.npz")
    parser.add_argument("--output-root", help="Checkpoint root supplied by an orchestrated run")
    parser.add_argument("--run-id")
    parser.add_argument("--runs-root", default=ROOT / "runs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", type=int, default=0)
    args = parser.parse_args()

    if args.output_root:
        checkpoint_root = Path(args.output_root)
    else:
        run_dir = prepare_run_dir("full-refit", args.run_id, args.runs_root, args.force)
        checkpoint_root = run_dir / "checkpoints"
        print(f"Run directory: {run_dir}")
    run_full_refit(
        checkpoint_root,
        args.data_path,
        force=args.force,
        force_cpu=args.cpu,
        dcn_epochs=args.quick or 40,
        k25_epochs=args.quick or 40,
        k15_epochs=args.quick or 60,
        graph_epochs=args.quick or 40,
    )


if __name__ == "__main__":
    main()
