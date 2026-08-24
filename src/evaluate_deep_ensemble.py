import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models import DeepConvNet, EEGNet82, GraphEEGNet
from src.extract import EEGDataset
from src.train import load_and_split_data_pipeline


CHECKPOINTS = (
    # name, weight, checkpoint, optional input time slice
    ("eegnet_k25_seed42", 3, "best_eegnet_k25_seed42.pth", None),
    ("eegnet_k15_swa_seed42", 1, "eegnet_k15_swa_seed42_standalone.pth", None),
    ("eegnet_k15_swa_seed123", 2, "eegnet_k15_swa_seed123.pth", None),
    ("dcn_window_0_2000", 2, "dcn_window_0_2000_seed42.pth", (50, 551)),
    ("graph_eeg_seed42", 2, "graph_eeg_seed42.pth", (50, 551)),
)


def load_models(device):
    checkpoint_dir = os.path.join(ROOT, "models", "checkpoints")
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
        model.load_state_dict(torch.load(path, map_location=device))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(ROOT, "data", "processed", "eeg_dataset.npz")
    _, _, val_x, val_y, test_x, test_y, _, _ = load_and_split_data_pipeline(data_path)
    models = load_models(device)
    print("Models:", ", ".join(f"{name}={weight}" for name, weight, _, _ in models))
    print(f"Validation accuracy: {accuracy(models, val_x, val_y, device):.2f}%")
    print(f"Test accuracy: {accuracy(models, test_x, test_y, device):.2f}%")


if __name__ == "__main__":
    main()
