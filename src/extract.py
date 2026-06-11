import os
import scipy.io as sio
import numpy as np

# Try to import torch and scikit-learn to make sure they are documented and available
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Resolve the absolute project root directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)


def load_and_preprocess_eeg_data(mat_path):
    """
    Loads raw EEG handwritten imagery data from .mat format, transposes
    the dimensions to standard machine learning format, cleans types,
    and returns variables.
    
    Original structure inside .mat:
    - data: shape (24, 801, 7800) -> (electrodes, time_points, samples)
    - label: shape (7800, 1) -> labels 1 to 26 corresponding to alphabets A to Z
    - channel_labels: shape (24, 1) -> cell array containing string labels for the 24 electrodes
    - time_points: shape (801, 1) -> time in milliseconds (-200 ms to 3000 ms)
    
    Returns:
    --------
    data_transposed: np.ndarray, shape (7800, 24, 801)
        EEG data reshaped to (samples, channels, time_points)
    labels_1indexed: np.ndarray, shape (7800,)
        1-based class labels (1 to 26)
    labels_0indexed: np.ndarray, shape (7800,)
        0-based class labels (0 to 25)
    channels: list of str
        The 24 EEG channel names (e.g. ['Fp1', 'Fp2', ...])
    time_points: np.ndarray, shape (801,)
        Time values in ms for each sample point
    """
    print(f"Loading Matlab file from: {mat_path} ...")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Matlab file not found at: {mat_path}")
        
    mat = sio.loadmat(mat_path)
    
    # 1. Extract and transpose data to (samples, channels, time_points)
    raw_data = mat['data']  # (24, 801, 7800)
    data_transposed = np.transpose(raw_data, (2, 0, 1))  # (7800, 24, 801)
    
    # 2. Extract and flatten labels (originally shape (7800, 1))
    labels_1indexed = mat['label'].flatten()  # 1 to 26
    labels_0indexed = labels_1indexed - 1  # Convert to 0 to 25 for Python ML zero-indexing
    
    # 3. Clean channel labels cell array to a simple Python list of strings
    raw_channels = mat['channel_labels']  # (24, 1) cell array
    channels = [str(c[0][0]) if len(c) > 0 and len(c[0]) > 0 else f"Ch{i+1}" for i, c in enumerate(raw_channels)]
    
    # 4. Flatten time points
    time_points = mat['time_points'].flatten()  # (801,)
    
    print("Data extraction complete!")
    print(f"  - Extracted EEG data shape: {data_transposed.shape} (samples, channels, time_points)")
    print(f"  - Unique labels: {len(np.unique(labels_0indexed))} classes (0 to 25)")
    print(f"  - Number of channels: {len(channels)} {channels}")
    print(f"  - Sampling parameters: {len(time_points)} time points spanning from {time_points[0]}ms to {time_points[-1]}ms")
    
    return data_transposed, labels_1indexed, labels_0indexed, channels, time_points


def save_as_numpy_archive(output_path, data, labels_1indexed, labels_0indexed, channels, time_points):
    """
    Saves extracted arrays to a compressed .npz archive for lightning-fast loading in Python.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        data=data,
        labels_1indexed=labels_1indexed,
        labels_0indexed=labels_0indexed,
        channels=np.array(channels, dtype=str),
        time_points=time_points
    )
    print(f"Saved compressed NumPy dataset to: {output_path}")


def get_alphabet_letter(label_1indexed):
    """
    Returns the character representing the 1-based class label.
    """
    return chr(ord('A') + label_1indexed - 1)


# -----------------------------------------------------------------------------
# PyTorch Dataset Definition
# -----------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class EEGDataset(Dataset):
        """
        Custom PyTorch Dataset for EEG Handwriting Imagery signals.
        Treated as a 2D single-channel image, adding a channel dimension
        to match PyTorch CNN shape format: (Batch, Channel, Height, Width)
        i.e., (Batch, 1, EEG_Channels, Time_Points) -> (N, 1, 24, 801)
        """
        def __init__(self, data, labels, transform=None):
            # Input data is shape (N, 24, 801) -> reshape to (N, 1, 24, 801)
            self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
            # Labels should be class indices (0 to 25)
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.labels[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
else:
    class EEGDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available. Please install torch to use the EEGDataset class.")


# -----------------------------------------------------------------------------
# Main pipeline execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Standard structured paths
    mat_path = os.path.join(ROOT_DIR, "data", "raw", "data_EEG_AI.mat")
    npz_path = os.path.join(ROOT_DIR, "data", "processed", "eeg_dataset.npz")


    # 1. Run extractor
    data, labels_1indexed, labels_0indexed, channels, time_points = load_and_preprocess_eeg_data(mat_path)
    
    # 2. Save NumPy Archive
    save_as_numpy_archive(npz_path, data, labels_1indexed, labels_0indexed, channels, time_points)
    

    # 3. Demonstrate Stratified Train-Test Split if scikit-learn is available
    if SKLEARN_AVAILABLE:
        print("\nDemonstrating Stratified Train-Test Split (80% Train, 20% Test) using scikit-learn...")
        X_train, X_test, y_train, y_test = train_test_split(
            data, 
            labels_0indexed, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels_0indexed
        )
        print(f"  - Training set size: {X_train.shape} with labels {y_train.shape}")
        print(f"  - Testing set size: {X_test.shape} with labels {y_test.shape}")
        
        # Verify class balance
        classes, counts = np.unique(y_train, return_counts=True)
        print(f"  - Class balance check (Train): Each of the {len(classes)} classes has {counts[0]} samples")
        
        # PyTorch integration demo
        if TORCH_AVAILABLE:
            print("\nInstantiating PyTorch Datasets and DataLoader...")
            train_dataset = EEGDataset(X_train, y_train)
            test_dataset = EEGDataset(X_test, y_test)
            
            # Create a simple loader
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Fetch a single batch to demonstrate format
            batch_x, batch_y = next(iter(train_loader))
            print(f"  - PyTorch DataLoader batch input shape: {batch_x.shape} -> (Batch, Channels, Height, Width)")
            print(f"  - PyTorch DataLoader batch label shape: {batch_y.shape}")
            print("  - Ready for CNN model training!")
            

