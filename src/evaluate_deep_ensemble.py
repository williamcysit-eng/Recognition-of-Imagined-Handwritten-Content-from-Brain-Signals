import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models import DeepConvNet, EEGNet82, GraphEEGNet
from src.checkpoint_manifest import LEGACY_FIVE_MODEL_CHECKPOINTS
from src.extract import EEGDataset
from src.train import load_and_split_data_pipeline


CHECKPOINTS = LEGACY_FIVE_MODEL_CHECKPOINTS


def load_models(device, checkpoint_dir=None):
    checkpoint_dir = checkpoint_dir or os.path.join(ROOT, "models", "checkpoints")
    models = []
    for name, weight, filename, time_slice in CHECKPOINTS:
        if name.startswith("graph"):
            model = GraphEEGNet(24, 26)
        elif name.startswith("dcn"):
            time_points = 801 if time_slice is None else time_slice[1] - time_slice[0]
            model = DeepConvNet(24, 26, time_points, temporal_kernel=15, dropout_rate=0.5)
        else:
            kernel = 25 if "k25" in name else 15
            model = EEGNet82(24, 26, input_time_points=801,
                             temporal_kernel_length=kernel, dropout_rate=0.3)
        path = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required checkpoint is missing: {path}")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device).eval()
        models.append((name, weight, model, time_slice))
    return models


def accuracy(models, x, y, device):
    correct = total = 0
    loader = DataLoader(EEGDataset(x, y), batch_size=128, shuffle=False)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = 0
            for _, weight, model, time_slice in models:
                model_x = batch_x if time_slice is None else batch_x[..., time_slice[0]:time_slice[1]]
                logits = logits + weight * model(model_x)
            correct += (logits.argmax(1) == batch_y).sum().item()
            total += batch_y.numel()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate the historical fixed five-model ensemble")
    parser.add_argument("--checkpoint-dir", default=os.path.join(ROOT, "models", "checkpoints"))
    parser.add_argument("--data-path", default=os.path.join(ROOT, "data", "processed", "eeg_dataset.npz"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    _, _, val_x, val_y, test_x, test_y, _, _ = load_and_split_data_pipeline(args.data_path)
    models = load_models(device, args.checkpoint_dir)
    print("Models:", ", ".join(f"{name}={weight}" for name, weight, _, _ in models))
    print(f"Validation accuracy: {accuracy(models, val_x, val_y, device):.2f}%")
    print(f"Test accuracy: {accuracy(models, test_x, test_y, device):.2f}%")


if __name__ == "__main__":
    main()
