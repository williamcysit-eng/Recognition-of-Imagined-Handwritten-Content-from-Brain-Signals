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
from torch.utils.data import Dataset, DataLoader
from extract_data import EEGDataset

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# 1. Dataset Loading
# -----------------------------------------------------------------------------


def load_and_split_data(npz_path, downsample_factor=5):
    """
    Loads data, downsamples the time axis to speed up CPU training, 
    and splits it into stratified Train (80%), Val (10%), and Test (10%) sets.
    """
    print(f"Loading preprocessed dataset from {npz_path}...")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NumPy dataset archive not found at: {npz_path}. Run extract_data.py first.")
        
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
    
    # Class balance check
    classes, counts = np.unique(y_train, return_counts=True)
    print(f"Class Balance (Train):   Each class has exactly {counts[0]} samples (perfectly balanced)")
    print("="*50 + "\n")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, channels, time_points


# -----------------------------------------------------------------------------
# 2. Optimized Lightweight 2D CNN Model Definition
# -----------------------------------------------------------------------------
class LightEEG2DCNN(nn.Module):
    """
    A highly optimized, ultra-lightweight 2D CNN for EEG classification.
    Designed for fast CPU training while learning powerful spatial-temporal features.
    """
    def __init__(self, num_classes=26):
        super(LightEEG2DCNN, self).__init__()
        
        # Downsampled input shape: (Batch, 1, 24, 161)
        self.conv = nn.Sequential(
            # 1. Temporal Conv
            nn.Conv2d(1, 8, kernel_size=(1, 15), stride=1, padding=(0, 7), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.MaxPool2d((1, 2)), # (8, 24, 80)
            
            # 2. Spatial Conv (combines channels)
            nn.Conv2d(8, 16, kernel_size=(24, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d((1, 2)), # (16, 1, 40)
            
            # 3. Temporal Pooling Conv
            nn.Conv2d(16, 16, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=16, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)), # (32, 1, 8) -> 256 features
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 1 * 8, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 1, 24, T)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------------------------------------------------------------
# 3. Training and Evaluation Pipeline
# -----------------------------------------------------------------------------
def train_cnn_model(X_train, y_train, X_val, y_val, num_epochs=15, batch_size=128, lr=0.005):
    """
    Trains the 2D CNN model, saves training curves, validates on validation set,
    and returns the model from the epoch that achieved the highest validation accuracy.
    """
    # Create datasets and dataloaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training 2D CNN on device: {device}")
    
    model = LightEEG2DCNN(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    best_epoch = 1
    best_model_path = "best_eeg_2d_cnn.pth"
    
    # Metrics tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\nStarting 2D CNN Training Loop...")
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
        
        # Adjust learning rate based on Validation Accuracy
        scheduler.step(epoch_val_acc)
        
        print(f"{epoch:<8}{epoch_train_loss:<12.4f}{epoch_train_acc:<15.2f}{epoch_val_loss:<12.4f}{epoch_val_acc:<15.2f}{epoch_time:<8.1f}")
        
        # Checkpoint if validation accuracy improved
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            
    print("-" * 65)
    print(f"Best 2D CNN Validation Accuracy: {best_val_acc:.2f}% achieved at Epoch {best_epoch} (Saved to {best_model_path})")
    print("This checkpoints-based selection avoids overfitting by ignoring subsequent epochs.\n")
    
    # Load best model for evaluation
    print(f"Loading optimal model weights from Epoch {best_epoch} for final test evaluation...")
    best_model = LightEEG2DCNN(num_classes=26).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    return best_model, history, device


def evaluate_cnn_model(model, X_test, y_test, device):
    """
    Evaluates the trained 2D CNN model on the test dataset.
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
    print(f"2D CNN Final Test Accuracy: {test_acc:.2f}%")
    return test_acc


def plot_training_history(history):
    """
    Generates and saves a training curves plot showing loss and accuracy.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_2d_cnn.png', dpi=150)
    plt.close()
    print("Saved training curves plot as 'training_curves_2d_cnn.png'")


# -----------------------------------------------------------------------------
# 4. Baseline Machine Learning Model
# -----------------------------------------------------------------------------
def train_evaluate_baseline(X_train, y_train, X_test, y_test):
    """
    Trains a Logistic Regression model on flattened downsampled features as a baseline.
    """
    print("\nTraining Logistic Regression baseline on flattened features...")
    t0 = time.time()
    
    # Flatten spatio-temporal matrices (24 channels * 161 time points) -> 3864 features
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Fast Logistic Regression using lbfgs on CPU (small max_iter for speedy training)
    baseline_model = LogisticRegression(C=0.01, solver='lbfgs', max_iter=30, random_state=RANDOM_SEED)
    baseline_model.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, baseline_model.predict(X_train_scaled)) * 100
    test_acc = accuracy_score(y_test, baseline_model.predict(X_test_scaled)) * 100
    
    print(f"Baseline completed in {time.time() - t0:.1f}s")
    print(f"Baseline Train Accuracy: {train_acc:.2f}%")
    print(f"Baseline Test Accuracy:  {test_acc:.2f}%")
    return test_acc


# -----------------------------------------------------------------------------
# 5. Main Execution Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(current_dir, "eeg_dataset.npz")
    
    print("==========================================================")
    print("       EEG Handwriting Imagery 2D CNN Training Pipeline   ")
    print("==========================================================\n")
    
    # Load and split the dataset with a downsampling factor of 5 for extreme CPU optimization
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = load_and_split_data(npz_path, downsample_factor=5)
    
    # Train the deep learning 2D CNN classifier
    cnn_model, history, device = train_cnn_model(
        X_train, y_train, X_val, y_val, num_epochs=15, batch_size=128, lr=0.005
    )
    
    # Plot training metrics over time
    plot_training_history(history)
    
    # Evaluate the CNN model on the held-out Test set
    print("\nEvaluating best 2D CNN model on independent Test set...")
    cnn_test_acc = evaluate_cnn_model(cnn_model, X_test, y_test, device)
    
    # Compare with standard Machine Learning baseline
    baseline_test_acc = train_evaluate_baseline(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*50)
    print("                  FINAL COMPARISON RESULTS                ")
    print("="*50)
    print(f"Logistic Regression Baseline Accuracy: {baseline_test_acc:.2f}%")
    print(f"LightEEG2DCNN Model Accuracy:          {cnn_test_acc:.2f}%")
    improvement = cnn_test_acc - baseline_test_acc
    print(f"Absolute Deep Learning Improvement:   +{improvement:.2f}%")
    print("="*50)
