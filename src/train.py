import hashlib
import json
import os
import sys
import time
import argparse
import numpy as np

# Dynamic import configuration: Add root directory to python path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import models and custom dataset
from models import EEGNet82, apply_max_norm_constraints, DeepConvNet, EEGInception
from src.extract import EEGDataset

# Try to import PyTorch and scikit-learn
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch is not available. Deep learning models cannot be trained.")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn is not available. Data splitting and baselines cannot be run.")

# Set random seed for reproducibility
RANDOM_SEED = 42

def set_seed(seed):
    global RANDOM_SEED
    RANDOM_SEED = seed
    np.random.seed(RANDOM_SEED)
    if TORCH_AVAILABLE:
        torch.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def select_device(force_cpu=False):
    """Choose CUDA, then Apple Metal, with CPU as the portable fallback."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


SPLIT_MANIFEST_VERSION = 1
SPLIT_POLICY = "classwise_index_block_v1"
DEFAULT_SPLIT_MANIFEST_PATH = os.path.join(ROOT_DIR, "data", "split_manifest.json")


def _fingerprint_array(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _dataset_identity(data, labels, channels, time_points):
    return {
        "data_shape": list(data.shape),
        "data_dtype": str(data.dtype),
        "data_fingerprint": _fingerprint_array(data),
        "labels_shape": list(labels.shape),
        "labels_dtype": str(labels.dtype),
        "labels_fingerprint": _fingerprint_array(labels),
        "channels": [str(channel) for channel in channels],
        "channels_fingerprint": _fingerprint_array(channels),
        "time_points_shape": list(time_points.shape),
        "time_points_dtype": str(time_points.dtype),
        "time_points_fingerprint": _fingerprint_array(time_points),
    }


def _find_acquisition_metadata(dataset):
    known_keys = {
        "data",
        "labels_1indexed",
        "labels_0indexed",
        "channels",
        "time_points",
    }
    # Treat every additional archive field as provenance requiring inspection.
    return tuple(sorted(key for key in dataset.files if key not in known_keys))


def _build_classwise_split_indices(labels):
    """Build the documented stored-index split without claiming chronology."""
    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        n_samples = len(indices)
        n_train = int(0.80 * n_samples)
        n_val = int(0.10 * n_samples)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])
    return tuple(
        np.asarray(indices, dtype=np.int64)
        for indices in (train_idx, val_idx, test_idx)
    )


def _validate_partition_indices(partitions, sample_count):
    combined = np.concatenate(partitions)
    expected = np.arange(sample_count, dtype=np.int64)
    if (
        combined.size != sample_count
        or np.unique(combined).size != sample_count
        or not np.array_equal(np.sort(combined), expected)
    ):
        raise RuntimeError("Split manifest partitions are not disjoint and exhaustive")


def _build_split_manifest(data, labels, channels, time_points, partitions, metadata_keys):
    _validate_partition_indices(partitions, len(labels))
    names = ("train", "validation", "test")
    class_counts = {}
    for label in np.unique(labels):
        class_counts[str(int(label))] = {
            name: int(np.sum(labels[indices] == label))
            for name, indices in zip(names, partitions)
        }
    return {
        "manifest_version": SPLIT_MANIFEST_VERSION,
        "split_policy": SPLIT_POLICY,
        "dataset": _dataset_identity(data, labels, channels, time_points),
        "partitions": {
            f"{name}_indices": indices.tolist()
            for name, indices in zip(names, partitions)
        },
        "partition_counts": {
            name: int(len(indices)) for name, indices in zip(names, partitions)
        },
        "class_counts": class_counts,
        "acquisition_metadata": {
            "available": bool(metadata_keys),
            "keys": list(metadata_keys),
            "status": (
                "unavailable_index_block_approximation"
                if not metadata_keys
                else "present_but_not_group_split"
            ),
        },
    }


def write_split_manifest(npz_path, manifest_path=None):
    """Write a frozen manifest after verifying the archive's available metadata."""
    manifest_path = manifest_path or DEFAULT_SPLIT_MANIFEST_PATH
    with np.load(npz_path, allow_pickle=False) as dataset:
        raw_data = dataset["data"].astype(np.float32)
        labels = dataset["labels_0indexed"]
        channels = dataset["channels"].copy()
        time_points = dataset["time_points"].copy()
        metadata_keys = _find_acquisition_metadata(dataset)
    if metadata_keys:
        raise RuntimeError(
            "Additional archive fields are present "
            f"({', '.join(metadata_keys)}), but no verified group-aware splitter "
            "is configured; refusing to create an index-only manifest."
        )
    partitions = _build_classwise_split_indices(labels)
    manifest = _build_split_manifest(
        raw_data, labels, channels, time_points, partitions, metadata_keys
    )
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    print(f"Wrote frozen split manifest to {manifest_path}")
    print(
        "Acquisition/session metadata unavailable; "
        "the manifest records a stored-index approximation."
    )
    return manifest


def _load_frozen_split_manifest(
    manifest_path, data, labels, channels, time_points, metadata_keys
):
    if metadata_keys:
        raise RuntimeError(
            "Additional archive fields are present "
            f"({', '.join(metadata_keys)}), but no verified group-aware splitter "
            "is configured; refusing to use a class-wise index split."
        )
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            "Frozen split manifest not found at "
            f"{manifest_path}. Run with --write-split-manifest once, inspect it, "
            "and keep it with the dataset."
        )
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("manifest_version") != SPLIT_MANIFEST_VERSION:
        raise RuntimeError("Unsupported frozen split manifest version")
    if manifest.get("split_policy") != SPLIT_POLICY:
        raise RuntimeError("Frozen split manifest uses an unexpected split policy")
    expected_identity = _dataset_identity(data, labels, channels, time_points)
    if manifest.get("dataset") != expected_identity:
        raise RuntimeError(
            "Dataset identity does not match the frozen split manifest; "
            "refusing to change test membership silently."
        )
    recorded_metadata = manifest.get("acquisition_metadata", {})
    if tuple(recorded_metadata.get("keys", ())) != metadata_keys:
        raise RuntimeError("Acquisition metadata changed after split manifest creation")
    names = ("train", "validation", "test")
    manifest_partitions = tuple(
        np.asarray(
            manifest["partitions"][f"{name}_indices"],
            dtype=np.int64,
        )
        for name in names
    )
    _validate_partition_indices(manifest_partitions, len(labels))
    expected_partitions = _build_classwise_split_indices(labels)
    if any(
        not np.array_equal(actual, expected)
        for actual, expected in zip(manifest_partitions, expected_partitions)
    ):
        raise RuntimeError(
            "Frozen split manifest does not match the documented class-wise "
            "index-block policy."
        )
    print(f"Frozen split manifest verified: {manifest_path}")
    print(
        "Acquisition/session metadata unavailable; "
        "using a stored-index approximation rather than verified chronology."
    )
    return manifest_partitions


set_seed(42)

# -----------------------------------------------------------------------------
# DEFAULT PIPELINE CONFIGURATIONS (Change these to easily tune default runs)
# -----------------------------------------------------------------------------
DEFAULT_EPOCHS = 150
DEFAULT_MIXUP = True
DEFAULT_MIXUP_ALPHA = 0.2

# -----------------------------------------------------------------------------
# 1. Loading and Splitting Data
# -----------------------------------------------------------------------------
def load_and_split_data_pipeline(
    npz_path,
    downsample_factor=1,
    include_test=True,
    split_manifest_path=None,
):
    """
    Load data using the verified frozen split manifest.

    The current archive has no acquisition/session metadata, so its documented
    class-wise stored-index split is an explicit approximation rather than a
    verified chronological or group-preserving split.
    """
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be a positive integer")
    print(f"Loading preprocessed dataset from {npz_path}...")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"NumPy dataset archive not found at: {npz_path}. Run extract.py first."
        )

    with np.load(npz_path, allow_pickle=False) as dataset:
        raw_data = dataset["data"].astype(np.float32)  # (7800, 24, 801)
        labels = dataset["labels_0indexed"]
        channels_array = dataset["channels"].copy()
        time_points = dataset["time_points"].copy()
        metadata_keys = _find_acquisition_metadata(dataset)
    channels = list(channels_array)
    manifest_path = split_manifest_path or DEFAULT_SPLIT_MANIFEST_PATH
    train_idx, val_idx, test_idx = _load_frozen_split_manifest(
        manifest_path,
        raw_data,
        labels,
        channels_array,
        time_points,
        metadata_keys,
    )

    # Downsample time points to accelerate CPU calculations after split
    # identity and frozen membership have been verified.
    if downsample_factor > 1:
        print(
            "Downsampling temporal dimension by factor of "
            f"{downsample_factor} for efficient CPU execution..."
        )
        data = raw_data[:, :, ::downsample_factor]
        time_points = time_points[::downsample_factor]
    else:
        data = raw_data

    print("Using frozen class-wise index-block split.")
    print(
        f"  - Train: {len(train_idx)} | Validation: {len(val_idx)} | "
        f"Test: {len(test_idx)}"
    )
    X_train, y_train = data[train_idx], labels[train_idx]
    X_val, y_val = data[val_idx], labels[val_idx]
    if include_test:
        X_test, y_test = data[test_idx], labels[test_idx]
    else:
        X_test, y_test = None, None

    return X_train, y_train, X_val, y_val, X_test, y_test, channels, time_points


# -----------------------------------------------------------------------------
# 2. General Training Loop
# -----------------------------------------------------------------------------
def train_deep_learning_model(model_type, X_train, y_train, X_val, y_val, 
                              channels_count, time_points_count, num_epochs=15, 
                              batch_size=64, lr=0.005, temporal_kernel=64,
                              use_mixup=False, mixup_alpha=0.2, noise_std=0.0,
                              use_swa=False, swa_start_epoch=30,
                              force_cpu=False, quick_epochs=0,
                              cpu_threads=None):
    """
    Trains a deep learning model with validation-based checkpointing.
    """
    if quick_epochs > 0:
        num_epochs = quick_epochs
    device = select_device(force_cpu)
    is_cpu = device.type == "cpu"
    print(f"\nInitializing {model_type.upper()} on device: {device}")
    
    if is_cpu:
        torch.backends.mkldnn.enabled = True
        num_threads = cpu_threads if cpu_threads is not None else (max(1, os.cpu_count() - 2) if os.cpu_count() else 4)
        torch.set_num_threads(num_threads)
        print(f"CPU optimizations: MKL-DNN enabled, {num_threads} threads")
    
    if use_mixup:
        print(f"Applying Mixup Augmentation (alpha={mixup_alpha}) during training...")
    if use_swa:
        print(f"SWA enabled: averaging weights from epoch {swa_start_epoch}")
    
    if model_type == "deep_conv_net":
        dcn_kernel = 15  # 60ms at 250Hz
        model = DeepConvNet(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            temporal_kernel=dcn_kernel,
            dropout_rate=0.5
        ).to(device)
        best_model_path = os.path.join(ROOT_DIR, "models", "checkpoints", "best_deep_conv_net.pth")
    elif model_type == "eegnet":
        model = EEGNet82(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            temporal_kernel_length=temporal_kernel,
            dropout_rate=0.3
        ).to(device)
        best_model_path = os.path.join(ROOT_DIR, "models", "checkpoints", "best_eegnet.pth")
    elif model_type == "eeg_inception":
        model = EEGInception(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            dropout_rate=0.3
        ).to(device)
        best_model_path = os.path.join(ROOT_DIR, "models", "checkpoints", "best_eeg_inception.pth")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if is_cpu:
        model = model.to(memory_format=torch.channels_last)
    
    # Datasets and Loaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    dl_kwargs = dict(num_workers=2, persistent_workers=True) if is_cpu else {}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **dl_kwargs)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0 if model_type == "deep_conv_net" else 0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5, 
        patience=3,
        min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    best_epoch = 1
    
    # Early Stopping config
    patience = 40
    epochs_no_improve = 0
    
    # SWA tracking
    swa_state_dict = None
    swa_n = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\nStarting {model_type.upper()} Training Loop...")
    print("-" * 70)
    print(f"{'Epoch':<8}{'Train Loss':<12}{'Train Acc (%)':<15}{'Val Loss':<12}{'Val Acc (%)':<15}{'Time (s)':<8}")
    print("-" * 70)
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if is_cpu:
                batch_x = batch_x.to(memory_format=torch.channels_last)
            
            # Gaussian noise augmentation for EEG signals
            if noise_std > 0:
                batch_x = batch_x + torch.randn_like(batch_x) * noise_std
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_mixup and mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                batch_size_curr = batch_x.size(0)
                index = torch.randperm(batch_size_curr, device=device)
                
                mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
                outputs = model(mixed_x)
                
                loss = lam * criterion(outputs, batch_y) + (1 - lam) * criterion(outputs, batch_y[index])
                
                _, predicted = outputs.max(1)
                correct_train += lam * predicted.eq(batch_y).sum().item() + (1 - lam) * predicted.eq(batch_y[index]).sum().item()
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                _, predicted = outputs.max(1)
                correct_train += predicted.eq(batch_y).sum().item()
                
            loss.backward()
            optimizer.step()
            
            # EEGNet-specific Max-Norm constraints step
            if model_type == "eegnet":
                apply_max_norm_constraints(model)
                
            train_loss += loss.item() * batch_x.size(0)
            total_train += batch_y.size(0)
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(batch_y).sum().item()
                total_val += batch_y.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100
        epoch_time = time.time() - t0
        
        # Log history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)
        
        # SWA weight averaging (start after warmup)
        if use_swa and epoch >= swa_start_epoch:
            if swa_state_dict is None:
                swa_state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}
                swa_n = 1
            else:
                for k in swa_state_dict:
                    swa_state_dict[k] = (swa_state_dict[k] * swa_n + model.state_dict()[k].detach()) / (swa_n + 1)
                swa_n += 1
        
        print(f"{epoch:<8}{epoch_train_loss:<12.4f}{epoch_train_acc:<15.2f}{epoch_val_loss:<12.4f}{epoch_val_acc:<15.2f}{epoch_time:<8.1f}")
        
        # Validation loss checkpointing & Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered! No improvement in validation loss for {patience} consecutive epochs.")
            break
            
    print("-" * 70)
    if use_swa and swa_state_dict is not None:
        print(f"Best {model_type.upper()} Validation Loss: {best_val_loss:.4f} (Accuracy: {history['val_acc'][best_epoch-1]:.2f}%) achieved at Epoch {best_epoch}")
        print(f"Using SWA model (averaged {swa_n} checkpoints from epoch {swa_start_epoch} to {epoch})...")
    else:
        print(f"Best {model_type.upper()} Validation Loss: {best_val_loss:.4f} (Accuracy: {history['val_acc'][best_epoch-1]:.2f}%) achieved at Epoch {best_epoch}")
        print(f"Loading optimal model weights from Epoch {best_epoch} (prevents overfitting)...")
    
    # Reload optimal model weights or use SWA
    if model_type == "deep_conv_net":
        dcn_kernel = 15
        best_model = DeepConvNet(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            temporal_kernel=dcn_kernel,
            dropout_rate=0.5
        ).to(device)
    elif model_type == "eegnet":
        best_model = EEGNet82(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            temporal_kernel_length=temporal_kernel,
            dropout_rate=0.3
        ).to(device)
    elif model_type == "eeg_inception":
        best_model = EEGInception(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            dropout_rate=0.3
        ).to(device)
    
    if use_swa and swa_state_dict is not None:
        best_model.load_state_dict(swa_state_dict)
    else:
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    return best_model, history, device


# -----------------------------------------------------------------------------
# 3. Model Testing
# -----------------------------------------------------------------------------
def evaluate_model_on_split(
    model_type, model, X_eval, y_eval, device, split_name="development"
):
    eval_dataset = EEGDataset(X_eval, y_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
    accuracy = (correct / total) * 100
    print(f"\n--- {model_type.upper()} {split_name.upper()} EVALUATION METRICS ---")
    print(f"  * Single-Trial {split_name.title()} Accuracy: {accuracy:.2f}%")
    return accuracy


def evaluate_ensemble_on_split(
    model_a,
    model_b,
    X_eval,
    y_eval,
    device,
    split_name="development",
    name_a="DCN",
    name_b="EEGNet",
):
    eval_dataset = EEGDataset(X_eval, y_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    model_a.eval()
    model_b.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = (model_a(batch_x) + model_b(batch_x)) / 2.0
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
    accuracy = (correct / total) * 100
    print(f"\n--- ENSEMBLE ({name_a} + {name_b}) {split_name.upper()} EVALUATION ---")
    print(f"  * Single-Trial {split_name.title()} Accuracy: {accuracy:.2f}%")
    return accuracy


def evaluate_gated_ensemble_on_split(
    model_a,
    model_b,
    X_eval,
    y_eval,
    device,
    split_name="development",
    name_a="DCN",
    name_b="EEGNet",
):
    eval_dataset = EEGDataset(X_eval, y_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    model_a.eval()
    model_b.eval()
    correct, total, agree_count = 0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out_a = model_a(batch_x)
            out_b = model_b(batch_x)
            probs_a = out_a.softmax(dim=1)
            probs_b = out_b.softmax(dim=1)
            pred_a = out_a.argmax(dim=1)
            pred_b = out_b.argmax(dim=1)
            agree = pred_a == pred_b
            agree_count += agree.sum().item()
            use_a = probs_a.max(dim=1).values >= probs_b.max(dim=1).values
            final_pred = torch.where(agree, pred_a, torch.where(use_a, pred_a, pred_b))
            correct += final_pred.eq(batch_y).sum().item()
            total += batch_y.size(0)
    accuracy = (correct / total) * 100
    agreement_rate = (agree_count / total) * 100
    print(f"\n--- GATED ENSEMBLE ({name_a} + {name_b}) {split_name.upper()} EVALUATION ---")
    print(
        f"  * Single-Trial {split_name.title()} Accuracy: {accuracy:.2f}% "
        f"(agreement rate: {agreement_rate:.1f}%)"
    )
    return accuracy


def evaluate_ensemble_3_fixed_on_split(
    dcn_model,
    eeg_model,
    ei_model,
    X_eval,
    y_eval,
    device,
    split_name="development",
    w_dcn=5.0,
    w_eeg=5.0,
    w_ei=1.0,
):
    weights = np.asarray((w_dcn, w_eeg, w_ei), dtype=np.float64)
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("ensemble weights must be finite, non-negative, and non-zero")
    weights /= weights.sum()
    eval_dataset = EEGDataset(X_eval, y_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    dcn_model.eval()
    eeg_model.eval()
    ei_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = (
                weights[0] * dcn_model(batch_x)
                + weights[1] * eeg_model(batch_x)
                + weights[2] * ei_model(batch_x)
            )
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
    accuracy = (correct / total) * 100
    print(
        f"\n--- 3-MODEL FIXED-WEIGHT ENSEMBLE {split_name.upper()} EVALUATION ---"
    )
    print(
        f"  * Fixed weights: DCN={weights[0]:.3f}, "
        f"EEGNet={weights[1]:.3f}, EI={weights[2]:.3f}"
    )
    print(f"  * Single-Trial {split_name.title()} Accuracy: {accuracy:.2f}%")
    return accuracy


# -----------------------------------------------------------------------------
# 4. Baseline Machine Learning Classifier
# -----------------------------------------------------------------------------
def run_logistic_regression_baseline(
    X_train, y_train, X_eval, y_eval, split_name="development"
):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_eval_flat = X_eval.reshape(X_eval.shape[0], -1)

    baseline_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, C=0.05, random_state=RANDOM_SEED),
    )
    baseline_model.fit(X_train_flat, y_train)
    accuracy = accuracy_score(y_eval, baseline_model.predict(X_eval_flat)) * 100
    print(f"\n--- LOGISTIC REGRESSION {split_name.upper()} EVALUATION ---")
    print(f"  * Single-Trial {split_name.title()} Accuracy: {accuracy:.2f}%")
    return accuracy


# 4. Pipeline Orchestrator Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified EEG Handwriting Imagery Classification Pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["deep_conv_net", "eegnet", "eeg_inception", "ensemble", "all"],
        default="deep_conv_net",
        help="Model architecture to train (default: deep_conv_net)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsampling factor for time series (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs to train (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--no-mixup",
        action="store_true",
        default=not DEFAULT_MIXUP,
        help="Disable Mixup data augmentation during training",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian noise augmentation (default: 0.0 = off)",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=DEFAULT_MIXUP_ALPHA,
        help=f"Alpha parameter for Beta distribution in Mixup (default: {DEFAULT_MIXUP_ALPHA})",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU training (for timing benchmarks)",
    )
    parser.add_argument(
        "--quick",
        type=int,
        default=0,
        help="Limit to N epochs for timing estimates (0 = full training)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Enable cudnn.benchmark for ~47%% faster GPU training (minor accuracy trade-off)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    evaluation_group = parser.add_mutually_exclusive_group()
    evaluation_group.add_argument(
        "--development-only",
        dest="development_only",
        action="store_true",
        help="Train and evaluate only on training/development partitions (default)",
    )
    evaluation_group.add_argument(
        "--evaluate-test",
        dest="development_only",
        action="store_false",
        help="Explicitly evaluate the fixed held-out test partition",
    )
    parser.set_defaults(development_only=True)
    parser.add_argument(
        "--split-manifest",
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help="Path to the frozen split manifest",
    )
    parser.add_argument(
        "--write-split-manifest",
        action="store_true",
        help="Write a frozen split manifest and exit without training",
    )
    args = parser.parse_args()

    if args.downsample < 1:
        parser.error("--downsample must be a positive integer")
    if args.quick < 0:
        parser.error("--quick must be non-negative")

    set_seed(args.seed)

    if args.fast and TORCH_AVAILABLE and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(
            "--fast mode enabled: cudnn.benchmark=True "
            "(~47% faster epochs, minor accuracy trade-off)"
        )

    if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
        print(
            "Missing required libraries. Please run "
            "'pip install scikit-learn torch numpy scipy matplotlib'"
        )
        sys.exit(1)

    npz_path = os.path.join(ROOT_DIR, "data", "processed", "eeg_dataset.npz")
    if args.write_split_manifest:
        write_split_manifest(npz_path, args.split_manifest)
        sys.exit(0)

    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = (
        load_and_split_data_pipeline(
            npz_path,
            downsample_factor=args.downsample,
            include_test=not args.development_only,
            split_manifest_path=args.split_manifest,
        )
    )
    if args.development_only:
        X_eval, y_eval, evaluation_name = X_val, y_val, "development"
        print("\nDevelopment-only mode: the fixed test partition was not materialized for evaluation.")
    else:
        X_eval, y_eval, evaluation_name = X_test, y_test, "test"
        print("\nExplicit final-test mode: evaluating the fixed held-out test partition.")

    channels_count = X_train.shape[1]
    time_points_count = X_train.shape[2]

    # Scale temporal kernel based on downsampling factor to keep it aligned
    # with BCI dynamics (~half the default 250 Hz sampling rate).
    effective_sr = 250 / args.downsample
    temporal_kernel_len = int(effective_sr / 2)
    if temporal_kernel_len % 2 == 0:
        temporal_kernel_len += 1

    results = {}

    models_to_train = []
    if args.model == "deep_conv_net" or args.model == "all":
        models_to_train.append("deep_conv_net")
    if args.model == "eegnet" or args.model == "all":
        models_to_train.append("eegnet")
    if args.model == "eeg_inception" or args.model == "all":
        models_to_train.append("eeg_inception")

    for model_type in models_to_train:
        model, history, device = train_deep_learning_model(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            channels_count=channels_count,
            time_points_count=time_points_count,
            num_epochs=args.epochs,
            batch_size=64,
            lr=0.005,
            temporal_kernel=temporal_kernel_len,
            use_mixup=not args.no_mixup,
            mixup_alpha=args.mixup_alpha,
            noise_std=args.noise_std,
            quick_epochs=args.quick,
        )
        results[model_type] = evaluate_model_on_split(
            model_type,
            model,
            X_eval,
            y_eval,
            device,
            split_name=evaluation_name,
        )

    if args.model in ("ensemble", "all"):
        print("\n" + "=" * 60)
        print("  Training Ensemble: DeepConvNet + EEGNet")
        print("=" * 60)
        if args.cpu:
            print("Launching parallel CPU training (DCN + EEGNet in parallel threads)...")
            from concurrent.futures import ThreadPoolExecutor

            parallel_threads = (
                max(1, (os.cpu_count() - 2) // 2) if os.cpu_count() else 2
            )

            def _train_dcn():
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                return train_deep_learning_model(
                    model_type="deep_conv_net",
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    channels_count=channels_count,
                    time_points_count=time_points_count,
                    num_epochs=args.epochs,
                    batch_size=64,
                    lr=0.005,
                    temporal_kernel=temporal_kernel_len,
                    use_mixup=False,
                    mixup_alpha=0.2,
                    noise_std=0.0,
                    force_cpu=True,
                    quick_epochs=args.quick,
                    cpu_threads=parallel_threads,
                )

            def _train_eeg():
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                return train_deep_learning_model(
                    model_type="eegnet",
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    channels_count=channels_count,
                    time_points_count=time_points_count,
                    num_epochs=args.epochs,
                    batch_size=64,
                    lr=0.005,
                    temporal_kernel=15,
                    use_mixup=True,
                    mixup_alpha=0.2,
                    noise_std=0.07,
                    use_swa=True,
                    swa_start_epoch=25,
                    force_cpu=True,
                    quick_epochs=args.quick,
                    cpu_threads=parallel_threads,
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                dcn_future = executor.submit(_train_dcn)
                eeg_future = executor.submit(_train_eeg)
                dcn_model, dcn_history, _ = dcn_future.result()
                eeg_model, eeg_history, device = eeg_future.result()

            device = torch.device("cpu")
            ckpt_dir = os.path.join(ROOT_DIR, "models", "checkpoints")
            dcn_model = DeepConvNet(
                num_channels=channels_count,
                num_classes=26,
                input_time_points=time_points_count,
                temporal_kernel=15,
                dropout_rate=0.5,
            ).to(device)
            dcn_model.load_state_dict(
                torch.load(
                    os.path.join(ckpt_dir, "best_deep_conv_net.pth"),
                    map_location=device,
                )
            )
            eeg_model = EEGNet82(
                num_channels=channels_count,
                num_classes=26,
                input_time_points=time_points_count,
                temporal_kernel_length=15,
                dropout_rate=0.3,
            ).to(device)
            eeg_model.load_state_dict(
                torch.load(
                    os.path.join(ckpt_dir, "best_eegnet.pth"),
                    map_location=device,
                )
            )
        else:
            dcn_model, dcn_history, device = train_deep_learning_model(
                model_type="deep_conv_net",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                channels_count=channels_count,
                time_points_count=time_points_count,
                num_epochs=args.epochs,
                batch_size=64,
                lr=0.005,
                temporal_kernel=temporal_kernel_len,
                use_mixup=False,
                mixup_alpha=0.2,
                noise_std=0.0,
                quick_epochs=args.quick,
            )
            eeg_model, eeg_history, _ = train_deep_learning_model(
                model_type="eegnet",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                channels_count=channels_count,
                time_points_count=time_points_count,
                num_epochs=args.epochs,
                batch_size=64,
                lr=0.005,
                temporal_kernel=15,
                use_mixup=True,
                mixup_alpha=0.2,
                noise_std=0.07,
                use_swa=True,
                swa_start_epoch=25,
                quick_epochs=args.quick,
            )

        results["deep_conv_net"] = evaluate_model_on_split(
            "deep_conv_net",
            dcn_model,
            X_eval,
            y_eval,
            device,
            split_name=evaluation_name,
        )
        results["eegnet"] = evaluate_model_on_split(
            "eegnet",
            eeg_model,
            X_eval,
            y_eval,
            device,
            split_name=evaluation_name,
        )
        results["ensemble_dcn_eegnet"] = evaluate_ensemble_on_split(
            dcn_model,
            eeg_model,
            X_eval,
            y_eval,
            device,
            split_name=evaluation_name,
        )

        eeg_k25_model, _, _ = train_deep_learning_model(
            model_type="eegnet",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            channels_count=channels_count,
            time_points_count=time_points_count,
            num_epochs=args.epochs,
            batch_size=64,
            lr=0.005,
            temporal_kernel=25,
            use_mixup=True,
            mixup_alpha=0.2,
            noise_std=0.07,
            use_swa=False,
            swa_start_epoch=25,
            quick_epochs=args.quick,
        )
        results["eegnet_k25"] = evaluate_model_on_split(
            "eegnet_k25",
            eeg_k25_model,
            X_eval,
            y_eval,
            device,
            split_name=evaluation_name,
        )
        results["ensemble_3_k25"] = evaluate_ensemble_3_fixed_on_split(
            dcn_model,
            eeg_model,
            eeg_k25_model,
            X_eval,
            y_eval,
            device,
            split_name=evaluation_name,
            w_dcn=5,
            w_eeg=5,
            w_ei=1,
        )

    results["logistic_regression"] = run_logistic_regression_baseline(
        X_train,
        y_train,
        X_eval,
        y_eval,
        split_name=evaluation_name,
    )

    print(f"\nFINAL {evaluation_name.upper()} COMPARISON RESULTS")
    for model_name, accuracy in results.items():
        print(
            f"  * {model_name.upper():<25} "
            f"{evaluation_name.title()} Accuracy: {accuracy:.2f}%"
        )
