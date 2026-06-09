import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Dynamic import configuration: Add root directory to python path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import models and custom dataset
from models import LightEEG2DCNN, EEGNet82, apply_max_norm_constraints, EEGConformer
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
    from sklearn.model_selection import train_test_split
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
np.random.seed(RANDOM_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# 1. Loading and Splitting Data
# -----------------------------------------------------------------------------
def load_and_split_data_pipeline(npz_path, downsample_factor=5):
    """
    Loads preprocessed data and splits it into stratified Train (80%), Val (10%), and Test (10%).
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
        
    # Stratified Train-Test Split (80% Train, 20% Temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
    )
    
    # Stratified Val-Test Split from Temp (50% Val, 50% Test of the 20% -> 10% and 10%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print("\n" + "="*50)
    print("               DATASET SPLIT DETAILS              ")
    print("="*50)
    print(f"Total dataset size:      {data.shape[0]} samples")
    print(f"Train split size:        {X_train.shape[0]} samples ({X_train.shape[0]/data.shape[0]*100:.1f}%)")
    print(f"Validation split size:   {X_val.shape[0]} samples ({X_val.shape[0]/data.shape[0]*100:.1f}%)")
    print(f"Test split size:         {X_test.shape[0]} samples ({X_test.shape[0]/data.shape[0]*100:.1f}%)")
    print(f"EEG Signal Input Shape:  {X_train.shape[1:]} (Channels: {X_train.shape[1]}, Time Points: {X_train.shape[2]})")
    print(f"Number of classes:       {len(np.unique(labels))} (A to Z)")
    print(f"Class Balance (Train):   Each class has exactly {np.sum(y_train == 0)} samples (perfectly balanced)")
    print("="*50 + "\n")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, channels, time_points


def evaluate_multi_trial_averaging(model, X, y, device, group_size=5):
    """
    Groups trials of the same letter (class) together in chunks of `group_size` (3 to 5),
    calculates the average probability vector (softmax output) across the group,
    and returns the averaged classification accuracy.
    """
    model.eval()
    unique_classes = np.unique(y)
    total_groups = 0
    correct_groups = 0
    
    with torch.no_grad():
        for c in unique_classes:
            # Get all indices for class c
            indices = np.where(y == c)[0]
            num_trials = len(indices)
            
            # Chunk indices into groups of group_size
            for i in range(0, num_trials, group_size):
                group_idx = indices[i:i + group_size]
                if len(group_idx) < group_size:
                    # Skip incomplete last group to ensure equal averaging weight
                    continue
                
                # Fetch trials in the group and convert to PyTorch tensor
                trials = X[group_idx]  # Shape: (group_size, channels, time_points)
                trials_tensor = torch.tensor(trials, dtype=torch.float32).unsqueeze(1).to(device)
                
                # Forward pass
                logits = model(trials_tensor)  # Shape: (group_size, num_classes)
                probs = torch.softmax(logits, dim=1)  # Get probability vectors
                
                # Average probability vector across the group
                mean_probs = torch.mean(probs, dim=0)  # Shape: (num_classes,)
                
                # Predicted class
                pred_class = torch.argmax(mean_probs).item()
                
                if pred_class == c:
                    correct_groups += 1
                total_groups += 1
                
    accuracy = (correct_groups / total_groups) * 100 if total_groups > 0 else 0.0
    return accuracy


# -----------------------------------------------------------------------------
# 2. General Training Loop
# -----------------------------------------------------------------------------
def train_deep_learning_model(model_type, X_train, y_train, X_val, y_val, 
                              channels_count, time_points_count, num_epochs=15, 
                              batch_size=128, lr=0.005, temporal_kernel=64):
    """
    Trains either the 2D CNN or the fully-fledged EEGNet model, enforcing 
    checkpoint saving based on the highest validation accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing {model_type.upper()} on device: {device}")
    
    if model_type == "2d_cnn":
        model = LightEEG2DCNN(num_classes=26).to(device)
        best_model_path = os.path.join(ROOT_DIR, "models", "checkpoints", "best_eeg_2d_cnn.pth")
    elif model_type == "eegnet":
        model = EEGNet82(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            temporal_kernel_length=temporal_kernel,
            dropout_rate=0.3
        ).to(device)
        best_model_path = os.path.join(ROOT_DIR, "models", "checkpoints", "best_eegnet.pth")
    elif model_type == "conformer":
        model = EEGConformer(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            dropout_rate=0.3
        ).to(device)
        best_model_path = os.path.join(ROOT_DIR, "models", "checkpoints", "best_conformer.pth")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Datasets and Loaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02 if model_type == "2d_cnn" else 0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',  # Monitor validation loss (minimize)
        factor=0.5, 
        patience=3,  # Wait 3 epochs before reducing learning rate
        min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    best_epoch = 1
    
    # Early Stopping config
    patience = 10
    epochs_no_improve = 0
    
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
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # EEGNet-specific Max-Norm constraints step
            if model_type == "eegnet":
                apply_max_norm_constraints(model)
                
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(batch_y).sum().item()
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
        
        # ReduceLROnPlateau scheduler step based on validation loss
        scheduler.step(epoch_val_loss)
        
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
    print(f"Best {model_type.upper()} Validation Loss: {best_val_loss:.4f} (Accuracy: {history['val_acc'][best_epoch-1]:.2f}%) achieved at Epoch {best_epoch}")
    print(f"Loading optimal model weights from Epoch {best_epoch} (prevents overfitting)...")
    
    # Reload optimal model weights
    if model_type == "2d_cnn":
        best_model = LightEEG2DCNN(num_classes=26).to(device)
    elif model_type == "eegnet":
        best_model = EEGNet82(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            temporal_kernel_length=temporal_kernel,
            dropout_rate=0.3
        ).to(device)
    elif model_type == "conformer":
        best_model = EEGConformer(
            num_channels=channels_count,
            num_classes=26,
            input_time_points=time_points_count,
            dropout_rate=0.3
        ).to(device)
        
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    return best_model, history, device


# -----------------------------------------------------------------------------
# 3. Model Testing and Metrics Plotting
# -----------------------------------------------------------------------------
def evaluate_model_on_test_set(model_type, model, X_test, y_test, device):
    """
    Evaluates the loaded model on the test dataset.
    Computes both single-trial accuracy and multi-trial evaluation averaging (group sizes 3, 4, 5).
    """
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
            
    test_acc = (correct / total) * 100
    print(f"\n--- {model_type.upper()} EVALUATION METRICS ---")
    print(f"  * Single-Trial Held-Out Test Accuracy: {test_acc:.2f}%")
    
    # Compute multi-trial averaging for group sizes 3, 4, 5
    for k in [3, 4, 5]:
        mt_acc = evaluate_multi_trial_averaging(model, X_test, y_test, device, group_size=k)
        print(f"  * Multi-Trial Averaged Accuracy (K={k}): {mt_acc:.2f}%")
        
    return test_acc


def plot_metrics_history(model_type, history):
    """
    Generates and saves the loss and accuracy metrics plots.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss')
    plt.title(f'{model_type.upper()} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy')
    plt.title(f'{model_type.upper()} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(ROOT_DIR, "outputs", "figures", f"training_curves_{model_type}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved {model_type.upper()} training curves plot as: {plot_path}")


# -----------------------------------------------------------------------------
# 4. Baseline Machine Learning Classifier
# -----------------------------------------------------------------------------
def run_logistic_regression_baseline(X_train, y_train, X_test, y_test):
    """
    Trains a Logistic Regression baseline model on flattened features using scikit-learn.
    """
    print("\nTraining Logistic Regression baseline using scikit-learn...")
    t0 = time.time()
    
    # Flatten spatio-temporal matrices (channels * time points)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Force the baseline solver to actually converge using pipeline with standard scaling
    baseline_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, C=0.05, random_state=RANDOM_SEED)
    )
    baseline_model.fit(X_train_flat, y_train)
    
    train_acc = accuracy_score(y_train, baseline_model.predict(X_train_flat)) * 100
    test_acc = accuracy_score(y_test, baseline_model.predict(X_test_flat)) * 100
    
    print(f"Baseline completed in {time.time() - t0:.1f}s")
    print(f"Baseline Train Accuracy: {train_acc:.2f}%")
    print(f"Baseline Test Accuracy:  {test_acc:.2f}%")
    return test_acc


# -----------------------------------------------------------------------------
# 5. Pipeline Orchestrator Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified EEG Handwriting Imagery Classification Pipeline")
    parser.add_argument("--model", type=str, choices=["2d_cnn", "eegnet", "conformer", "both"], default="2d_cnn",
                        help="Model architecture to train (default: 2d_cnn)")
    parser.add_argument("--downsample", type=int, default=5,
                        help="Downsampling factor for time series (default: 5 for fast CPU execution)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of epochs to train (default: 15)")
    args = parser.parse_args()
    
    print("==========================================================")
    print("     UNIFIED EEG IMAGERY ML/DL CLASSIFICATION PIPELINE    ")
    print("==========================================================\n")
    
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
    if args.model == "2d_cnn" or args.model == "both":
        models_to_train.append("2d_cnn")
    if args.model == "eegnet" or args.model == "both":
        models_to_train.append("eegnet")
    if args.model == "conformer" or args.model == "both":
        models_to_train.append("conformer")
        
    # Run Deep Learning models
    for model_type in models_to_train:
        # Train
        model, history, device = train_deep_learning_model(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            channels_count=channels_count,
            time_points_count=time_points_count,
            num_epochs=args.epochs,
            batch_size=128,
            lr=0.005,
            temporal_kernel=temporal_kernel_len
        )
        
        # Plot curves
        plot_metrics_history(model_type, history)
        
        # Evaluate on test set
        test_acc = evaluate_model_on_test_set(model_type, model, X_test, y_test, device)
        results[model_type] = test_acc
        
    # Run scikit-learn Baseline Classifier
    baseline_test_acc = run_logistic_regression_baseline(X_train, y_train, X_test, y_test)
    results["logistic_regression"] = baseline_test_acc
    
    # Print Final Comparison
    print("\n" + "="*50)
    print("                  FINAL COMPARISON RESULTS                ")
    print("="*50)
    for model_name, acc in results.items():
        print(f"  * {model_name.upper():<25} Test Accuracy: {acc:.2f}%")
    print("="*50 + "\n")
