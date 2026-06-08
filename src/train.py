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
from models import LightEEG2DCNN, EEGNet82, apply_max_norm_constraints, EEGJEPA, compute_sigreg_loss, EEGJEPAClassifier, EEGConformer
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
    print("-" * 65)
    print(f"{'Epoch':<8}{'Train Loss':<12}{'Train Acc (%)':<15}{'Val Loss':<12}{'Val Acc (%)':<15}{'Time (s)':<8}")
    print("-" * 65)
    
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
            
    print("-" * 65)
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


def train_jepa_model(X_train, y_train, X_val, y_val, channels_count, time_points_count, 
                     num_epochs=15, batch_size=128, lr=0.001, temporal_kernel=64,
                     reg_weight=0.1):
    """
    Trains the EEGJEPA self-supervised world model (next-embedding prediction + SIGReg).
    Saves the best model based on validation prediction loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing EEGJEPA on device: {device}")
    
    model = EEGJEPA(
        num_channels=channels_count,
        input_time_points=time_points_count,
        temporal_kernel_length=temporal_kernel,
        dropout_rate=0.3
    ).to(device)
    
    best_model_path = os.path.join(ROOT_DIR, "models", "checkpoints", "best_jepa.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Datasets and Loaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    best_epoch = 1
    
    history = {
        'train_loss': [], 'train_mse': [], 'train_reg': [],
        'val_loss': [], 'val_mse': [], 'val_reg': []
    }
    
    print("\nStarting EEGJEPA Training Loop...")
    print("-" * 80)
    print(f"{'Epoch':<8}{'Train Loss':<12}{'Train MSE':<12}{'Train Reg':<12}{'Val Loss':<12}{'Val MSE':<12}{'Time (s)':<8}")
    print("-" * 80)
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_reg = 0.0
        total_samples = 0
        
        for batch_x, _ in train_loader:  # Target class label is ignored
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            pred_embed, target_embed = model(batch_x)
            
            # Predictor loss (L2 MSE distance)
            mse = nn.functional.mse_loss(pred_embed, target_embed)
            
            # Regularization constraint (SIGReg on predicted embeddings to prevent collapse)
            reg = compute_sigreg_loss(pred_embed)
            
            loss = mse + reg_weight * reg
            
            loss.backward()
            optimizer.step()
            
            # Target Encoder EMA weight update
            model.update_target_ema()
            
            # Max-Norm constraints step for both Online and Target Encoders (reusing eegnet step)
            apply_max_norm_constraints(model.online_encoder)
            apply_max_norm_constraints(model.target_encoder)
            
            batch_size_curr = batch_x.size(0)
            train_loss += loss.item() * batch_size_curr
            train_mse += mse.item() * batch_size_curr
            train_reg += reg.item() * batch_size_curr
            total_samples += batch_size_curr
            
        epoch_train_loss = train_loss / total_samples
        epoch_train_mse = train_mse / total_samples
        epoch_train_reg = train_reg / total_samples
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_reg = 0.0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                pred_embed, target_embed = model(batch_x)
                
                mse = nn.functional.mse_loss(pred_embed, target_embed)
                reg = compute_sigreg_loss(pred_embed)
                loss = mse + reg_weight * reg
                
                batch_size_curr = batch_x.size(0)
                val_loss += loss.item() * batch_size_curr
                val_mse += mse.item() * batch_size_curr
                val_reg += reg.item() * batch_size_curr
                total_val_samples += batch_size_curr
                
        epoch_val_loss = val_loss / total_val_samples
        epoch_val_mse = val_mse / total_val_samples
        epoch_val_reg = val_reg / total_val_samples
        epoch_time = time.time() - t0
        
        # Log history
        history['train_loss'].append(epoch_train_loss)
        history['train_mse'].append(epoch_train_mse)
        history['train_reg'].append(epoch_train_reg)
        history['val_loss'].append(epoch_val_loss)
        history['val_mse'].append(epoch_val_mse)
        history['val_reg'].append(epoch_val_reg)
        
        scheduler.step(epoch_val_loss)
        
        print(f"{epoch:<8}{epoch_train_loss:<12.4f}{epoch_train_mse:<12.4f}{epoch_train_reg:<12.4f}{epoch_val_loss:<12.4f}{epoch_val_mse:<12.4f}{epoch_time:<8.1f}")
        
        # Validation loss checkpointing (lower is better)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            
    print("-" * 80)
    print(f"Best EEGJEPA Validation Loss: {best_val_loss:.4f} achieved at Epoch {best_epoch}")
    print(f"Loading optimal model weights from Epoch {best_epoch}...")
    
    # Reload optimal model weights
    best_model = EEGJEPA(
        num_channels=channels_count,
        input_time_points=time_points_count,
        temporal_kernel_length=temporal_kernel,
        dropout_rate=0.3
    ).to(device)
    
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    return best_model, history, device


def train_jepa_downstream_classifier(model_type, X_train, y_train, X_val, y_val,
                                     channels_count, time_points_count, num_epochs=15,
                                     batch_size=128, lr=0.005, temporal_kernel=64):
    """
    Loads pre-trained EEGJEPA weights, instantiates EEGJEPAClassifier, and trains 
    it for downstream letter classification (0 to 25).
    
    If model_type == "jepa_probe", the encoder weights are frozen (Linear Probing).
    If model_type == "jepa_finetune", the encoder weights are trainable (Fine-Tuning).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing Downstream Classifier ({model_type.upper()}) on device: {device}")
    
    # 1. Instantiate the JEPA architecture first to load checkpoints
    jepa_model = EEGJEPA(
        num_channels=channels_count,
        input_time_points=time_points_count,
        temporal_kernel_length=temporal_kernel,
        dropout_rate=0.3
    ).to(device)
    
    jepa_checkpoint = os.path.join(ROOT_DIR, "models", "checkpoints", "best_jepa.pth")
    if not os.path.exists(jepa_checkpoint):
        print(f"Pre-trained JEPA weights not found at: {jepa_checkpoint}")
        print("Automatically performing self-supervised JEPA pre-training first...")
        # Train JEPA first for half epochs to get the pre-trained weights
        jepa_model, _, _ = train_jepa_model(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            channels_count=channels_count, time_points_count=time_points_count,
            num_epochs=max(5, num_epochs // 2), batch_size=batch_size, lr=0.001,
            temporal_kernel=temporal_kernel
        )
    else:
        print(f"Loading optimal pre-trained JEPA weights from: {jepa_checkpoint}")
        jepa_model.load_state_dict(torch.load(jepa_checkpoint, map_location=device))
        
    # 2. Extract the pre-trained online encoder
    encoder = jepa_model.online_encoder
    
    # 3. Create the classifier
    freeze_encoder = (model_type == "jepa_probe")
    model = EEGJEPAClassifier(
        encoder=encoder,
        num_classes=26,
        freeze_encoder=freeze_encoder
    ).to(device)
    
    best_model_path = os.path.join(ROOT_DIR, "models", "checkpoints", f"best_{model_type}.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Classifier parameters: {total_params:,} (Trainable: {trainable_params:,})")
    
    # Datasets and Loaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    # If probe, we only optimize self.fc parameters. If finetune, we optimize everything.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    best_epoch = 1
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\nStarting {model_type.upper()} Downstream Training...")
    print("-" * 65)
    print(f"{'Epoch':<8}{'Train Loss':<12}{'Train Acc (%)':<15}{'Val Loss':<12}{'Val Acc (%)':<15}{'Time (s)':<8}")
    print("-" * 65)
    
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
            
            # Since our encoder wraps standard EEGNet layers, apply the max-norm constraint on the encoder part
            apply_max_norm_constraints(model.encoder)
                
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
        
        scheduler.step(epoch_val_acc)
        
        print(f"{epoch:<8}{epoch_train_loss:<12.4f}{epoch_train_acc:<15.2f}{epoch_val_loss:<12.4f}{epoch_val_acc:<15.2f}{epoch_time:<8.1f}")
        
        # Validation accuracy checkpointing
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            
    print("-" * 65)
    print(f"Best {model_type.upper()} Validation Accuracy: {best_val_acc:.2f}% achieved at Epoch {best_epoch}")
    print(f"Loading optimal model weights from Epoch {best_epoch}...")
    
    # Reload optimal model weights
    best_model = EEGJEPAClassifier(
        encoder=encoder,
        num_classes=26,
        freeze_encoder=freeze_encoder
    ).to(device)
    
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    return best_model, history, device


# -----------------------------------------------------------------------------
# 3. Model Testing and Metrics Plotting
# -----------------------------------------------------------------------------
def evaluate_model_on_test_set(model_type, model, X_test, y_test, device):
    """
    Evaluates the loaded model on the test dataset.
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
    print(f"{model_type.upper()} Final Held-Out Test Accuracy: {test_acc:.2f}%")
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


def evaluate_jepa_on_test_set(model, X_test, y_test, device, reg_weight=0.1):
    """
    Evaluates the trained JEPA model on the test dataset.
    """
    test_dataset = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            pred_embed, target_embed = model(batch_x)
            
            mse = nn.functional.mse_loss(pred_embed, target_embed)
            reg = compute_sigreg_loss(pred_embed)
            loss = mse + reg_weight * reg
            
            batch_size_curr = batch_x.size(0)
            total_loss += loss.item() * batch_size_curr
            total_mse += mse.item() * batch_size_curr
            total_samples += batch_size_curr
            
    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    print(f"EEGJEPA Final Held-Out Test Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f})")
    return avg_loss


def plot_jepa_metrics_history(history):
    """
    Generates and saves the loss and MSE metrics plots for EEGJEPA.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('EEGJEPA Training and Validation Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MSE curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_mse'], 'b-o', label='Training MSE')
    plt.plot(epochs, history['val_mse'], 'r-s', label='Validation MSE')
    plt.title('EEGJEPA Training and Validation Next-Embedding MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(ROOT_DIR, "outputs", "figures", "training_curves_jepa.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved EEGJEPA training curves plot as: {plot_path}")


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
    
    # Standard scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Fast Logistic Regression using lbfgs on CPU
    baseline_model = LogisticRegression(C=0.01, solver='lbfgs', max_iter=30, random_state=RANDOM_SEED)
    baseline_model.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, baseline_model.predict(X_train_scaled)) * 100
    test_acc = accuracy_score(y_test, baseline_model.predict(X_test_scaled)) * 100
    
    print(f"Baseline completed in {time.time() - t0:.1f}s")
    print(f"Baseline Train Accuracy: {train_acc:.2f}%")
    print(f"Baseline Test Accuracy:  {test_acc:.2f}%")
    return test_acc


# -----------------------------------------------------------------------------
# 5. Pipeline Orchestrator Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified EEG Handwriting Imagery Classification Pipeline")
    parser.add_argument("--model", type=str, choices=["2d_cnn", "eegnet", "conformer", "both", "jepa", "jepa_probe", "jepa_finetune"], default="2d_cnn",
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
    if args.model == "jepa":
        # JEPA self-supervised world model training
        jepa_model, history, device = train_jepa_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            channels_count=channels_count,
            time_points_count=time_points_count,
            num_epochs=args.epochs,
            batch_size=128,
            lr=0.001,
            temporal_kernel=temporal_kernel_len
        )
        
        plot_jepa_metrics_history(history)
        
        test_loss = evaluate_jepa_on_test_set(jepa_model, X_test, y_test, device)
        
        print("\n" + "="*50)
        print("                  JEPA PIPELINE COMPLETED                 ")
        print("="*50)
        print(f"  * EEGJEPA Final Held-Out Test Loss: {test_loss:.4f}")
        print("="*50 + "\n")
        sys.exit(0)
        
    elif args.model in ["jepa_probe", "jepa_finetune"]:
        # Pre-trained JEPA downstream classification training (linear probe or fine-tune)
        model, history, device = train_jepa_downstream_classifier(
            model_type=args.model,
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
        
        plot_metrics_history(args.model, history)
        
        test_acc = evaluate_model_on_test_set(args.model, model, X_test, y_test, device)
        
        print("\n" + "="*50)
        print("             JEPA DOWNSTREAM PIPELINE COMPLETED           ")
        print("="*50)
        print(f"  * {args.model.upper()} Final Held-Out Test Accuracy: {test_acc:.2f}%")
        print("="*50 + "\n")
        sys.exit(0)
        
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
