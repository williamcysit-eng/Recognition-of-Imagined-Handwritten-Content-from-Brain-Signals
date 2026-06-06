import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

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


def generate_visualizations(data, labels_1indexed, channels, time_points, figures_dir):
    """
    Plots and saves:
    1. Single trial EEG heatmap (channels x time points) for alphabet 'A'.
    2. Single trial EEG line plot (channels' voltage lines over time) for alphabet 'A'.
    3. Average EEG heatmap for alphabet 'A' across all its 300 trials.
    4. Average EEG line plot for alphabet 'A' across all its 300 trials.
    """
    print("\nGenerating data visualization plots to match the instructions' Figure 3...")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Index of trials corresponding to Alphabet 'A' (label = 1)
    indices_a = np.where(labels_1indexed == 1)[0]
    
    # 1. Single trial data (e.g., the very first trial of alphabet 'A')
    single_trial_idx = indices_a[0]
    single_trial_data = data[single_trial_idx]  # Shape: (24, 801)
    
    # 2. Average pattern of all trials of alphabet 'A'
    average_data_a = np.mean(data[indices_a], axis=0)  # Shape: (24, 801)
    
    # Set up matplotlib style
    plt.rcParams.update({'font.size': 10, 'figure.titlesize': 14})
    
    # Y-ticks helpers
    y_ticks_idx = np.arange(len(channels))
    
    # PLOT 1: Single Trial Heatmap (Channels x Time)
    plt.figure(figsize=(12, 6))
    plt.imshow(single_trial_data, aspect='auto', cmap='RdBu_r', 
               extent=[time_points[0], time_points[-1], len(channels)-0.5, -0.5])
    plt.colorbar(label='Voltage / Amplitude (μV)')
    plt.yticks(y_ticks_idx, reversed(channels))
    plt.gca().invert_yaxis()  # Standard electrode ordering Fp1 at the top
    plt.xlabel('Time (ms)')
    plt.ylabel('EEG Channels')
    plt.title("Figure 3(a): Single Trial EEG Heatmap (Alphabet 'A')")
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.6, label='Stimulus Onset')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'single_trial_heatmap_A.png'), dpi=150)
    plt.close()
    print("  - Saved 'single_trial_heatmap_A.png'")
    
    # PLOT 2: Single Trial Line Plot (All Channels)
    plt.figure(figsize=(12, 6))
    offset = np.max(np.abs(single_trial_data)) * 0.8  # vertical offset for visibility
    for i, chan_name in enumerate(channels):
        plt.plot(time_points, single_trial_data[i] - i * offset, label=chan_name if i < 10 else "", alpha=0.8)
    plt.xlabel('Time (ms)')
    plt.ylabel('EEG Channels with offset')
    plt.yticks(-np.arange(len(channels)) * offset, channels)
    plt.title("Figure 3(b): Single Trial EEG Waveforms (Alphabet 'A')")
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.6)
    plt.xlim(time_points[0], time_points[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'single_trial_lineplot_A.png'), dpi=150)
    plt.close()
    print("  - Saved 'single_trial_lineplot_A.png'")

    # PLOT 3: Averaged Heatmap (Channels x Time)
    plt.figure(figsize=(12, 6))
    plt.imshow(average_data_a, aspect='auto', cmap='RdBu_r', 
               extent=[time_points[0], time_points[-1], len(channels)-0.5, -0.5])
    plt.colorbar(label='Voltage / Amplitude (μV)')
    plt.yticks(y_ticks_idx, reversed(channels))
    plt.gca().invert_yaxis()
    plt.xlabel('Time (ms)')
    plt.ylabel('EEG Channels')
    plt.title(f"Figure 3(c): Averaged EEG Heatmap across {len(indices_a)} trials (Alphabet 'A')")
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.6, label='Stimulus Onset')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'average_heatmap_A.png'), dpi=150)
    plt.close()
    print("  - Saved 'average_heatmap_A.png'")
    
    # PLOT 4: Averaged Line Plot (All Channels)
    plt.figure(figsize=(12, 6))
    offset_avg = np.max(np.abs(average_data_a)) * 0.8
    for i, chan_name in enumerate(channels):
        plt.plot(time_points, average_data_a[i] - i * offset_avg, label=chan_name if i < 10 else "", alpha=0.8)
    plt.xlabel('Time (ms)')
    plt.ylabel('EEG Channels with offset')
    plt.yticks(-np.arange(len(channels)) * offset_avg, channels)
    plt.title(f"Figure 3(d): Averaged EEG Waveforms across {len(indices_a)} trials (Alphabet 'A')")
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.6)
    plt.xlim(time_points[0], time_points[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'average_lineplot_A.png'), dpi=150)
    plt.close()
    print("  - Saved 'average_lineplot_A.png'")


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
    figures_dir = os.path.join(ROOT_DIR, "outputs", "figures")
    
    print("==========================================================")
    print("         EEG Handwriting Imagery Extraction Pipeline      ")
    print("==========================================================\n")
    
    # 1. Run extractor
    data, labels_1indexed, labels_0indexed, channels, time_points = load_and_preprocess_eeg_data(mat_path)
    
    # 2. Save NumPy Archive
    save_as_numpy_archive(npz_path, data, labels_1indexed, labels_0indexed, channels, time_points)
    
    # 3. Generate visualization plots (recreates Figure 3 described in the instructions)
    generate_visualizations(data, labels_1indexed, channels, time_points, figures_dir)
    
    # 4. Demonstrate Stratified Train-Test Split if scikit-learn is available
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
            
    print("\nExtraction and pre-processing completed successfully!")
    print("==========================================================")
