import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from extract_data import EEGDataset

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# 1. Fully-Fledged EEGNet Model Definition (EEGNet-8,2)
# -----------------------------------------------------------------------------
class EEGNet82(nn.Module):
    """
    Standard EEGNet architecture as proposed in:
    "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces"
    (https://iopscience.iop.org/article/10.1088/1741-2552/aae5d2)
    
    Parameters:
    -----------
    num_channels: int
        Number of EEG electrodes (default: 24).
    num_classes: int
        Number of target classes (default: 26).
    F1: int
        Number of temporal filters (default: 8).
    D: int
        Depth multiplier for spatial filters (default: 2, yielding F2 = F1 * D = 16).
    F2: int
        Number of pointwise filters (default: 16).
    input_time_points: int
        Length of the time sequence (default: 401 for downsample_factor=2).
    temporal_kernel_length: int
        Size of temporal filters, scaled to sampling rate. Typically sampling_rate // 2.
        For 250 Hz (un-downsampled), 125 is optimal. For 125 Hz (downsampled by 2), 64 is optimal.
    """
    def __init__(self, num_channels=24, num_classes=26, F1=8, D=2, F2=16, 
                 input_time_points=401, temporal_kernel_length=64, dropout_rate=0.3):
        super(EEGNet82, self).__init__()
        
        self.num_channels = num_channels
        self.input_time_points = input_time_points
        
        # 1. Block 1: Temporal Conv -> Depthwise Spatial Conv
        self.temporal_conv = nn.Sequential(
            # Temporal convolution with long filter along time axis (1, kernel_length)
            nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel_length), 
                      padding=(0, temporal_kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # Depthwise Spatial Convolution (constrained to groups=F1 to isolate filters per channel)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # EEGNet uses average pooling
            nn.Dropout(dropout_rate)
        )
        
        # 2. Block 2: Separable Temporal Conv
        self.separable_conv = nn.Sequential(
            # Depthwise temporal convolution
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), 
                      groups=F1 * D, bias=False),
            # Pointwise convolution (mixes filters)
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # 3. Dense Classifier Head
        # Dynamically compute flattened features size to prevent dimension mismatches
        self.flat_features = self._get_fc_input_size()
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, num_classes)
        )

    def _get_fc_input_size(self):
        """
        Passes a dummy tensor through the convolutional layers to determine the correct 
        number of flattened features input to the Linear classifier.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.num_channels, self.input_time_points)
            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
            return x.numel()

    def forward(self, x):
        # x shape: (Batch, 1, channels, time_points)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = self.fc(x)
        return x


# -----------------------------------------------------------------------------
# 2. Max-Norm Weight Constraints
# -----------------------------------------------------------------------------
@torch.no_grad()
def apply_max_norm_constraints(model, max_norm_spatial=1.0, max_norm_fc=0.25):
    """
    Applies Max-Norm weight constraints after the optimizer step as described in the 
    original EEGNet publication. This bounds individual filters to prevent overfitting.
    """
    for name, param in model.named_parameters():
        # Constrain spatial depthwise weights across channels (dim=2)
        if 'spatial_conv.0.weight' in name:
            norms = torch.norm(param.data, p=2, dim=2, keepdim=True)
            desired = torch.clamp(norms, max=max_norm_spatial)
            param.data *= (desired / (norms + 1e-10))
            
        # Constrain Dense Classification layer weights (dim=1)
        elif 'fc.1.weight' in name:
            norms = torch.norm(param.data, p=2, dim=1, keepdim=True)
            desired = torch.clamp(norms, max=max_norm_fc)
            param.data *= (desired / (norms + 1e-10))


# -----------------------------------------------------------------------------
# 3. Training Loop with Checkpointing and Constraints
# -----------------------------------------------------------------------------
def train_eegnet_model(X_train, y_train, X_val, y_val, channels_count, time_points_count,
                       num_epochs=20, batch_size=128, lr=0.003, temporal_kernel=64):
    """
    Trains the EEGNet model, enforces Max-Norm constraints, performs validation checkpointing,
    and returns the best-performing iteration.
    """
    # Datasets & Loaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training EEGNet on device: {device}")
    
    # Initialize EEGNet-8,2
    model = EEGNet82(
        num_channels=channels_count,
        num_classes=26,
        input_time_points=time_points_count,
        temporal_kernel_length=temporal_kernel,
        dropout_rate=0.3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    best_epoch = 1
    best_model_path = "best_eegnet.pth"
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\nStarting EEGNet Training Loop (incorporating Max-Norm Constraints)...")
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
            
            # CRITICAL EEGNet Step: Apply Max-Norm constraints immediately after SGD/Adam step
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
        
        # Track metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Scheduler Step
        scheduler.step(epoch_val_acc)
        
        print(f"{epoch:<8}{epoch_train_loss:<12.4f}{epoch_train_acc:<15.2f}{epoch_val_loss:<12.4f}{epoch_val_acc:<15.2f}{epoch_time:<8.1f}")
        
        # Checkpoint validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            
    print("-" * 65)
    print(f"Best EEGNet Validation Accuracy: {best_val_acc:.2f}% achieved at Epoch {best_epoch}")
    
    # Load optimal checkpoint weights
    print(f"Loading optimal model weights from Epoch {best_epoch} for test evaluation...")
    best_model = EEGNet82(
        num_channels=channels_count,
        num_classes=26,
        input_time_points=time_points_count,
        temporal_kernel_length=temporal_kernel,
        dropout_rate=0.3
    ).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    return best_model, history, device


# -----------------------------------------------------------------------------
# 4. Evaluation and Visualization
# -----------------------------------------------------------------------------
def evaluate_eegnet(model, X_test, y_test, device):
    """
    Evaluates the trained EEGNet model on the independent test set.
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
    print(f"EEGNet Final Test Accuracy: {test_acc:.2f}%")
    return test_acc


def plot_eegnet_history(history):
    """
    Generates and saves the loss and accuracy metrics plots.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('EEGNet Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy')
    plt.title('EEGNet Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_eegnet.png', dpi=150)
    plt.close()
    print("Saved EEGNet training curves plot as 'training_curves_eegnet.png'")


# -----------------------------------------------------------------------------
# 5. Main Execution Pipeline
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from train_2d_cnn import load_and_split_data
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(current_dir, "eeg_dataset.npz")
    
    print("==========================================================")
    print("      FULLY FLEDGED EEGNET-8,2 MODEL TRAINING PIPELINE    ")
    print("==========================================================\n")
    
    # Downsample by 2 is optimal for CPU: It yields 125 Hz sampling rate, 401 time points, 
    # capturing full motor imagery bands up to 62.5 Hz, while remaining computationally light.
    DOWNSAMPLE_FACTOR = 2
    
    # Load and split stratified data
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = load_and_split_data(
        npz_path, downsample_factor=DOWNSAMPLE_FACTOR
    )
    
    # Define channel and temporal counts based on loaded data
    channels_count = X_train.shape[1]      # Should be 24
    time_points_count = X_train.shape[2]  # Should be 401 (or 801 if DOWNSAMPLE_FACTOR=1)
    
    # Set temporal kernel to ~half of the sampling rate
    # For downsampling factor = 2 (125 Hz), 64 is optimal. 
    # For downsampling factor = 1 (250 Hz), 125 is optimal.
    temporal_kernel_len = 64 if DOWNSAMPLE_FACTOR == 2 else 125
    
    # Train the EEGNet model (20 epochs to show convergence)
    eegnet_model, history, device = train_eegnet_model(
        X_train=X_train, 
        y_train=y_train, 
        X_val=X_val, 
        y_val=y_val, 
        channels_count=channels_count, 
        time_points_count=time_points_count,
        num_epochs=20, 
        batch_size=128, 
        lr=0.003,
        temporal_kernel=temporal_kernel_len
    )
    
    # Plot and save curves
    plot_eegnet_history(history)
    
    # Evaluate test performance
    print("\nEvaluating fully-fledged EEGNet model on independent Test set...")
    eegnet_test_acc = evaluate_eegnet(eegnet_model, X_test, y_test, device)
    
    print("\n" + "="*50)
    print("                  TRAINING PROCESS COMPLETE               ")
    print("="*50)
    print(f"EEGNet configuration used: EEGNet-8,2 (F1=8, D=2, F2=16)")
    print(f"Time series resolution:   {time_points_count} points (downsampled by {DOWNSAMPLE_FACTOR})")
    print(f"Final Held-Out Test Acc:  {eegnet_test_acc:.2f}%")
    print("Checkpoints and weight bounds were applied dynamically.")
    print("="*50)
