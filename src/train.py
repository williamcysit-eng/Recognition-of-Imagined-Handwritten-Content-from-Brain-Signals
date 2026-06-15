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
RANDOM_SEED = 41
np.random.seed(RANDOM_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    using a class-wise chronological block split (preserving physical acquisition order).
    """
    print(f"Loading preprocessed dataset from {npz_path}...")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NumPy dataset archive not found at: {npz_path}. Run extract.py first.")
        
    dataset = np.load(npz_path, allow_pickle=False)
    raw_data = dataset['data']  # (7800, 24, 801)
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
        
    print("Performing Stratified Chronological Block Split (Class-wise Time Split)...")
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
                              use_swa=False, swa_start_epoch=30):
    """
    Trains a deep learning model with validation-based checkpointing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing {model_type.upper()} on device: {device}")
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
    
    # Datasets and Loaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
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
            
            # Gaussian noise augmentation for EEG signals
            if noise_std > 0:
                batch_x = batch_x + torch.randn_like(batch_x) * noise_std
            
            optimizer.zero_grad()
            
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

def evaluate_ensemble(model_a, model_b, X_test, y_test, device, name_a="DCN", name_b="EEGNet"):
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
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
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Standard deviation of Gaussian noise augmentation (default: 0.0 = off)")
    parser.add_argument("--mixup-alpha", type=float, default=DEFAULT_MIXUP_ALPHA,
                        help=f"Alpha parameter for Beta distribution in Mixup (default: {DEFAULT_MIXUP_ALPHA})")
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
        print("Missing required libraries. Please run 'pip install scikit-learn torch numpy scipy matplotlib'")
        sys.exit(1)
        
    npz_path = os.path.join(ROOT_DIR, "data", "processed", "eeg_dataset.npz")
    
    # Load and split stratified balanced data
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = load_and_split_data_pipeline(
        npz_path, downsample_factor=args.downsample
    )
    
    channels_count = X_train.shape[1]
    time_points_count = X_train.shape[2]
    
    # Scale temporal kernel based on downsampling factor to keep it aligned with BCI dynamics (~half SR)
    # default sampling rate is 250 Hz.
    effective_sr = 250 / args.downsample
    temporal_kernel_len = int(effective_sr / 2)
    if temporal_kernel_len % 2 == 0:
        temporal_kernel_len += 1  # must be odd for symmetric padding
        
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
        model, history, device = train_deep_learning_model(
            model_type=model_type,
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            channels_count=channels_count, time_points_count=time_points_count,
            num_epochs=args.epochs, batch_size=64, lr=0.005,
            temporal_kernel=temporal_kernel_len,
            use_mixup=not args.no_mixup, mixup_alpha=args.mixup_alpha,
            noise_std=args.noise_std,
        )
        test_acc = evaluate_model_on_test_set(model_type, model, X_test, y_test, device)
        results[model_type] = test_acc

    # Ensemble: DeepConvNet (clean) + EEGNet (mixup+noise)
    if args.model in ("ensemble", "all"):
        print("\n" + "=" * 60)
        print("  Training Ensemble: DeepConvNet + EEGNet")
        print("=" * 60)
        dcn_model, _, device = train_deep_learning_model(
            model_type="deep_conv_net",
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            channels_count=channels_count, time_points_count=time_points_count,
            num_epochs=args.epochs, batch_size=64, lr=0.005,
            temporal_kernel=temporal_kernel_len,
            use_mixup=False, mixup_alpha=0.2, noise_std=0.0,
        )
        eeg_model, _, _ = train_deep_learning_model(
            model_type="eegnet",
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            channels_count=channels_count, time_points_count=time_points_count,
            num_epochs=args.epochs, batch_size=64, lr=0.005,
            temporal_kernel=15,
            use_mixup=True, mixup_alpha=0.2, noise_std=0.07,
            use_swa=True, swa_start_epoch=25,
        )
        dcn_acc = evaluate_model_on_test_set("deep_conv_net", dcn_model, X_test, y_test, device)
        eeg_acc = evaluate_model_on_test_set("eegnet", eeg_model, X_test, y_test, device)
        results["deep_conv_net"] = dcn_acc
        results["eegnet"] = eeg_acc
        ens_acc = evaluate_ensemble(dcn_model, eeg_model, X_test, y_test, device)
        results["ensemble_dcn_eegnet"] = ens_acc
        
    # Run scikit-learn Baseline Classifier
    baseline_test_acc = run_logistic_regression_baseline(X_train, y_train, X_test, y_test)
    results["logistic_regression"] = baseline_test_acc
    
    # Print Final Comparison
    print("\nFINAL COMPARISON RESULTS")
    for model_name, acc in results.items():
        print(f"  * {model_name.upper():<25} Test Accuracy: {acc:.2f}%")
