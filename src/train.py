import os
import sys
import time
import argparse
import copy
import numpy as np

# Dynamic import configuration: Add root directory to python path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import models and custom dataset
from models import EEGNet82, apply_max_norm_constraints, DeepConvNet, EEGInception
from src.extract import EEGDataset
from src.run_utils import base_manifest, prepare_run_dir, save_torch_state, write_json

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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
def load_and_split_data_pipeline(npz_path, downsample_factor=1):
    """
    Loads preprocessed data and splits it into stratified Train (80%), Val (10%), and Test (10%)
    using a class-wise ordered block split. This preserves the order stored within
    each class; the archive does not contain a separate global acquisition timestamp.
    """
    print(f"Loading preprocessed dataset from {npz_path}...")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NumPy dataset archive not found at: {npz_path}. Run extract.py first.")

    dataset = np.load(npz_path, allow_pickle=False)
    raw_data = dataset['data'].astype(np.float32)  # (7800, 24, 801)
    labels = dataset['labels_0indexed']  # (7800,)
    channels = list(dataset['channels'])
    time_points = dataset['time_points']
    
    # Downsample time points to accelerate CPU calculations
    if downsample_factor > 1:
        print(f"Downsampling temporal dimension by factor of {downsample_factor} for efficient CPU execution...")
        data = raw_data[:, :, ::downsample_factor]
        time_points = time_points[::downsample_factor]
    else:
        data = raw_data
        
    print("Performing class-wise ordered block split (80/10/10)...")
    # Split each class individually chronologically (first 80% train, next 10% val, last 10% test)
    train_idx, val_idx, test_idx = [], [], []
    
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]  # Chronological indices for this class
        n_samples = len(indices)
        n_train = int(0.80 * n_samples)
        n_val = int(0.10 * n_samples)
        
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])
        
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    
    X_train, y_train = data[train_idx], labels[train_idx]
    X_val, y_val = data[val_idx], labels[val_idx]
    X_test, y_test = data[test_idx], labels[test_idx]
    
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
                              cpu_threads=None, early_stopping_patience=40,
                              channel_dropout=0.0, time_mask_samples=0,
                              amplitude_jitter=0.0):
    """
    Trains a deep learning model with validation-based checkpointing.
    """
    if quick_epochs > 0:
        num_epochs = quick_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if force_cpu:
        device = torch.device("cpu")
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
    elif model_type == "eegnet":
        model = EEGNet82(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            temporal_kernel_length=temporal_kernel,
            dropout_rate=0.3
        ).to(device)
    elif model_type == "eeg_inception":
        model = EEGInception(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            dropout_rate=0.3
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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
    best_state_dict = None
    
    # Early Stopping config
    patience = early_stopping_patience
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

            if amplitude_jitter > 0:
                scale = 1 + (2 * torch.rand(batch_x.size(0), 1, batch_x.size(2), 1,
                                            device=device) - 1) * amplitude_jitter
                batch_x = batch_x * scale

            if channel_dropout > 0:
                keep = (torch.rand(batch_x.size(0), 1, batch_x.size(2), 1,
                                   device=device) >= channel_dropout).to(batch_x.dtype)
                batch_x = batch_x * keep / max(1e-6, 1 - channel_dropout)

            if time_mask_samples > 0 and time_mask_samples < batch_x.size(-1):
                starts = torch.randint(0, batch_x.size(-1) - time_mask_samples + 1,
                                       (batch_x.size(0),), device=device)
                positions = torch.arange(batch_x.size(-1), device=device).view(1, 1, 1, -1)
                mask = (positions < starts.view(-1, 1, 1, 1)) | (
                    positions >= (starts + time_mask_samples).view(-1, 1, 1, 1))
                batch_x = batch_x * mask
            
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
                current_state = model.state_dict()
                for k in swa_state_dict:
                    if torch.is_floating_point(swa_state_dict[k]):
                        swa_state_dict[k] = (
                            swa_state_dict[k] * swa_n + current_state[k].detach()
                        ) / (swa_n + 1)
                    else:
                        swa_state_dict[k] = current_state[k].detach().clone()
                swa_n += 1
        
        print(f"{epoch:<8}{epoch_train_loss:<12.4f}{epoch_train_acc:<15.2f}{epoch_val_loss:<12.4f}{epoch_val_acc:<15.2f}{epoch_time:<8.1f}")
        
        # Validation loss checkpointing & Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
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
        if best_state_dict is None:
            raise RuntimeError("Training completed without producing a validation checkpoint")
        best_model.load_state_dict(best_state_dict)
    return best_model, history, device


# -----------------------------------------------------------------------------
# 3. Model Testing
# -----------------------------------------------------------------------------
def evaluate_model_on_test_set(model_type, model, X_test, y_test, device):
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            _, predicted = outputs.max(1)
            correct += predicted.eq(by).sum().item()
            total += by.size(0)
    test_acc = (correct / total) * 100
    print(f"\n--- {model_type.upper()} EVALUATION METRICS ---")
    print(f"  * Single-Trial Held-Out Test Accuracy: {test_acc:.2f}%")
    return test_acc

def adapt_bn_stats(model, loader, device):
    """Update BatchNorm statistics on unlabeled target data while keeping dropout frozen."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            model(bx)
    model.eval()

def evaluate_ensemble(model_a, model_b, X_test, y_test, device, name_a="DCN", name_b="EEGNet", adapt_bn=False):
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    if adapt_bn:
        adapt_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        adapt_bn_stats(model_a, adapt_loader, device)
        adapt_bn_stats(model_b, adapt_loader, device)
        print("  (BN adaptation applied)")
    model_a.eval()
    model_b.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out_a = model_a(bx)
            out_b = model_b(bx)
            outputs = (out_a + out_b) / 2.0
            _, predicted = outputs.max(1)
            correct += predicted.eq(by).sum().item()
            total += by.size(0)
    test_acc = (correct / total) * 100
    print(f"\n--- ENSEMBLE ({name_a} + {name_b}) EVALUATION ---")
    print(f"  * Single-Trial Held-Out Test Accuracy: {test_acc:.2f}%")
    return test_acc

def evaluate_ensemble_gated(model_a, model_b, X_test, y_test, device, name_a="DCN", name_b="EEGNet"):
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model_a.eval()
    model_b.eval()
    correct, total, agree_count = 0, 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out_a = model_a(bx)
            out_b = model_b(bx)
            probs_a = out_a.softmax(dim=1)
            probs_b = out_b.softmax(dim=1)
            pred_a = out_a.argmax(dim=1)
            pred_b = out_b.argmax(dim=1)
            conf_a = probs_a.max(dim=1).values
            conf_b = probs_b.max(dim=1).values
            agree = pred_a == pred_b
            agree_count += agree.sum().item()
            use_a = conf_a >= conf_b
            final_pred = torch.where(agree, pred_a, torch.where(use_a, pred_a, pred_b))
            correct += final_pred.eq(by).sum().item()
            total += by.size(0)
    test_acc = (correct / total) * 100
    print(f"\n--- GATED ENSEMBLE ({name_a} + {name_b}) EVALUATION ---")
    print(f"  * Single-Trial Held-Out Test Accuracy: {test_acc:.2f}% (agreement rate: {agree_count/total*100:.1f}%)")
    return test_acc

def evaluate_ensemble_3_fixed(dcn_model, eeg_model, ei_model, X_test, y_test, device,
                               w_dcn=5.0, w_eeg=5.0, w_ei=1.0, adapt_bn=True):
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    if adapt_bn:
        adapt_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        adapt_bn_stats(dcn_model, adapt_loader, device)
        adapt_bn_stats(eeg_model, adapt_loader, device)
        adapt_bn_stats(ei_model, adapt_loader, device)
    dcn_model.eval()
    eeg_model.eval()
    ei_model.eval()
    total_w = w_dcn + w_eeg + w_ei
    w_dcn /= total_w
    w_eeg /= total_w
    w_ei /= total_w
    print(f"\n--- 3-MODEL FIXED-WEIGHT ENSEMBLE ({w_dcn*11:.0f}:{w_eeg*11:.0f}:{w_ei*11:.0f}){' +BN-adapt' if adapt_bn else ''} ---")
    print(f"  * Fixed weights: DCN={w_dcn:.3f}, EEGNet={w_eeg:.3f}, EI={w_ei:.3f}")
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = w_dcn * dcn_model(bx) + w_eeg * eeg_model(bx) + w_ei * ei_model(bx)
            _, predicted = outputs.max(1)
            correct += predicted.eq(by).sum().item()
            total += by.size(0)
    test_acc = (correct / total) * 100
    print(f"  * Single-Trial Held-Out Test Accuracy: {test_acc:.2f}%")
    return test_acc



# -----------------------------------------------------------------------------
# 4. Baseline Machine Learning Classifier
# -----------------------------------------------------------------------------
def run_logistic_regression_baseline(X_train, y_train, X_test, y_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Force the baseline solver to actually converge using pipeline with standard scaling
    baseline_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, C=0.05, random_state=RANDOM_SEED)
    )
    baseline_model.fit(X_train_flat, y_train)
    test_acc = accuracy_score(y_test, baseline_model.predict(X_test_flat)) * 100
    return test_acc


# 4. Pipeline Orchestrator Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified EEG Handwriting Imagery Classification Pipeline")
    parser.add_argument("--model", type=str, choices=["deep_conv_net", "eegnet", "eeg_inception", "ensemble", "all"], default="deep_conv_net",
                        help="Model architecture to train (default: deep_conv_net)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsampling factor for time series (default: 1)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of epochs to train (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--no-mixup", action="store_true", default=not DEFAULT_MIXUP,
                        help="Disable Mixup data augmentation during training")
    parser.add_argument("--no-swa", action="store_true",
                        help="Disable SWA for a standalone EEGNet run")
    parser.add_argument("--noise-std", type=float, default=None,
                        help="Gaussian-noise standard deviation (model default: EEGNet=0.07, others=0.0)")
    parser.add_argument("--temporal-kernel", type=int, default=15,
                        help="EEGNet temporal-kernel length in samples (default: 15)")
    parser.add_argument("--mixup-alpha", type=float, default=DEFAULT_MIXUP_ALPHA,
                        help=f"Alpha parameter for Beta distribution in Mixup (default: {DEFAULT_MIXUP_ALPHA})")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Force CPU training (for timing benchmarks)")
    parser.add_argument("--quick", type=int, default=0,
                        help="Limit to N epochs for timing estimates (0 = full training)")
    parser.add_argument("--fast", action="store_true", default=False,
                        help="Enable cudnn.benchmark for ~47%% faster GPU training (minor accuracy trade-off)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--run-id", help="Run directory name (default: timestamped)")
    parser.add_argument("--runs-root", default=os.path.join(ROOT_DIR, "runs"),
                        help="Parent directory for isolated run outputs")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting files in an existing run directory")
    args = parser.parse_args()

    if args.downsample < 1:
        parser.error("--downsample must be at least 1")
    if args.temporal_kernel < 1:
        parser.error("--temporal-kernel must be positive")
    
    set_seed(args.seed)
    
    if args.fast and TORCH_AVAILABLE and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("--fast mode enabled: cudnn.benchmark=True (~47% faster epochs, minor accuracy trade-off)")
    
    if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
        print("Missing required libraries. Please run 'pip install scikit-learn torch numpy scipy matplotlib'")
        sys.exit(1)

    npz_path = os.path.join(ROOT_DIR, "data", "processed", "eeg_dataset.npz")
    if not os.path.exists(npz_path):
        parser.error(f"dataset not found: {npz_path}; run src/extract.py first")

    run_dir = prepare_run_dir("train", args.run_id, args.runs_root, args.force)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Load and split stratified balanced data
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = load_and_split_data_pipeline(
        npz_path, downsample_factor=args.downsample
    )
    
    channels_count = X_train.shape[1]
    time_points_count = X_train.shape[2]
    
    temporal_kernel_len = args.temporal_kernel
        
    # Store results for comparison
    results = {}
    
    # Determine models to run
    models_to_train = []
    if args.model == "deep_conv_net" or args.model == "all":
        models_to_train.append("deep_conv_net")
    if args.model == "eegnet" or args.model == "all":
        models_to_train.append("eegnet")
    if args.model == "eeg_inception" or args.model == "all":
        models_to_train.append("eeg_inception")
        
    # Run Deep Learning models
    for model_type in models_to_train:
        model_noise = (0.07 if model_type == "eegnet" else 0.0) if args.noise_std is None else args.noise_std
        model_mixup = model_type != "deep_conv_net" and not args.no_mixup
        model_swa = model_type == "eegnet" and not args.no_swa
        model, history, device = train_deep_learning_model(
            model_type=model_type,
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            channels_count=channels_count, time_points_count=time_points_count,
            num_epochs=args.epochs, batch_size=64, lr=0.005,
            temporal_kernel=temporal_kernel_len,
            use_mixup=model_mixup, mixup_alpha=args.mixup_alpha,
            noise_std=model_noise, force_cpu=args.cpu, quick_epochs=args.quick,
            use_swa=model_swa, swa_start_epoch=25,
        )
        save_torch_state(model, checkpoint_dir / f"single_{model_type}.pth", force=args.force)
        test_acc = evaluate_model_on_test_set(model_type, model, X_test, y_test, device)
        results[model_type] = test_acc

    # Ensemble: DeepConvNet (clean) + EEGNet (mixup+noise)
    if args.model in ("ensemble", "all"):
        print("\n" + "=" * 60)
        print("  Training Ensemble: DeepConvNet + EEGNet")
        print("=" * 60)
        dcn_model, dcn_history, device = train_deep_learning_model(
            model_type="deep_conv_net",
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            channels_count=channels_count, time_points_count=time_points_count,
            num_epochs=args.epochs, batch_size=64, lr=0.005,
            temporal_kernel=15, use_mixup=False, mixup_alpha=0.2, noise_std=0.0,
            force_cpu=args.cpu, quick_epochs=args.quick,
        )
        eeg_model, eeg_history, _ = train_deep_learning_model(
            model_type="eegnet",
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            channels_count=channels_count, time_points_count=time_points_count,
            num_epochs=args.epochs, batch_size=64, lr=0.005,
            temporal_kernel=15, use_mixup=True, mixup_alpha=0.2, noise_std=0.07,
            use_swa=True, swa_start_epoch=25, force_cpu=args.cpu,
            quick_epochs=args.quick,
        )
        save_torch_state(dcn_model, checkpoint_dir / "ensemble_dcn.pth", force=args.force)
        save_torch_state(eeg_model, checkpoint_dir / "ensemble_eegnet_k15_swa.pth", force=args.force)
        dcn_acc = evaluate_model_on_test_set("deep_conv_net", dcn_model, X_test, y_test, device)
        eeg_acc = evaluate_model_on_test_set("eegnet", eeg_model, X_test, y_test, device)
        results["deep_conv_net"] = dcn_acc
        results["eegnet"] = eeg_acc
        ens_acc = evaluate_ensemble(dcn_model, eeg_model, X_test, y_test, device)
        results["ensemble_dcn_eegnet"] = ens_acc
        eeg_k25_model, _, _ = train_deep_learning_model(
            model_type="eegnet",
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            channels_count=channels_count, time_points_count=time_points_count,
            num_epochs=args.epochs, batch_size=64, lr=0.005,
            temporal_kernel=25,
            use_mixup=True, mixup_alpha=0.2, noise_std=0.07,
            use_swa=False, swa_start_epoch=25,
            force_cpu=args.cpu, quick_epochs=args.quick,
        )
        save_torch_state(eeg_k25_model, checkpoint_dir / "ensemble_eegnet_k25.pth", force=args.force)
        eeg_k25_acc = evaluate_model_on_test_set("eegnet_k25", eeg_k25_model, X_test, y_test, device)
        results["eegnet_k25"] = eeg_k25_acc
        ens3_k25 = evaluate_ensemble_3_fixed(
            dcn_model, eeg_model, eeg_k25_model,
            X_test, y_test, device, w_dcn=5, w_eeg=5, w_ei=1, adapt_bn=False
        )
        results["ensemble_3_k25"] = ens3_k25
        
    # Run scikit-learn Baseline Classifier
    baseline_test_acc = run_logistic_regression_baseline(X_train, y_train, X_test, y_test)
    results["logistic_regression"] = baseline_test_acc
    
    # Print Final Comparison
    print("\nFINAL COMPARISON RESULTS")
    for model_name, acc in results.items():
        print(f"  * {model_name.upper():<25} Test Accuracy: {acc:.2f}%")

    manifest = base_manifest(run_dir.name, sys.argv, vars(args))
    manifest["results"] = results
    write_json(run_dir / "manifest.json", manifest, force=args.force)
    write_json(run_dir / "metrics.json", results, force=args.force)
    print(f"Saved run metadata to: {run_dir}")
