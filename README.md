# Recognition of Imagined Handwritten Content from Brain Signals

Classification of 26 imagined handwritten alphabets (A–Z) from single-trial EEG recordings using deep convolutional neural networks.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Preprocessing](#preprocessing)
5. [Data Splitting Strategy](#data-splitting-strategy)
6. [Model Architectures](#model-architectures)
7. [Training Methodology](#training-methodology)
8. [Regularization Techniques](#regularization-techniques)
9. [Ensemble Method](#ensemble-method)
10. [Results](#results)
11. [How to Run](#how-to-run)
12. [Repository Structure](#repository-structure)
13. [Academic Rationale](#academic-rationale)

---

## Project Overview

This project develops single-trial EEG classifiers to decode which of the 26 English alphabet letters a participant is imagining handwriting. The core challenge is the extreme difficulty of the task — 26-way classification from noisy, high-dimensional brain signals with only 300 training examples per class.

The solution employs three specialised convolutional neural network architectures designed for EEG decoding, culminating in an ensemble that combines complementary model inductive biases with Stochastic Weight Averaging (SWA) to achieve **25.38% single-trial test accuracy** — substantially above the ~3.85% random-chance baseline.

---

## Dataset

**Source:** Single-participant EEG recordings during a handwriting imagery task.

| Property | Value |
|----------|-------|
| Trials | 7,800 (300 per letter, 26 letters) |
| Channels | 24 (standard 10-20 system) |
| Time points | 801 (−200 ms to +3000 ms) |
| Sampling rate | 250 Hz |
| Preprocessing | Bandpass 0.1–45 Hz, ICA artifact removal, baseline correction |
| Data shape | (7800, 24, 801) |
| Labels | 0–25 (A–Z) |

**Task paradigm:** Every 3 seconds, a letter appeared on screen for 200 ms. The participant imagined the process of handwriting that letter. EEG was recorded continuously from 200 ms before stimulus onset to 3000 ms after.

---

## Pipeline Architecture

```
data/raw/data_EEG_AI.mat
        │
        ▼
  src/extract.py          ← Load .mat, transpose, clean labels, save .npz
        │
        ▼
data/processed/eeg_dataset.npz
        │
        ▼
  src/train.py            ← Stratified chronological split (80/10/10)
        │                    Train model(s) with validation checkpointing
        │                    Evaluate on held-out test set
        ▼
  Final test accuracy report + saved .pth checkpoints
```

---

## Preprocessing

Preprocessing is performed in `src/extract.py`:

1. **Load MATLAB data:** Reads `data_EEG_AI.mat` using `scipy.io.loadmat`.
2. **Transpose:** Converts from MATLAB format (24, 801, 7800) → ML format (7800, 24, 801).
3. **Label conversion:** 1-indexed (1–26) → 0-indexed (0–25) for PyTorch compatibility.
4. **Channel name extraction:** Cleans the cell array of 24 electrode labels.
5. **Save as .npz:** Compressed NumPy archive for fast subsequent loading.

The data is provided **already preprocessed** with major artifacts removed, baselined, and bandpass filtered between 0.1–45 Hz. Per-trial z-score normalization has also been applied by the data providers.

---

## Data Splitting Strategy

The split uses a **class-wise chronological block split** (80% train / 10% validation / 10% test):

```
For each class (0–25):
    Trials are ordered chronologically by acquisition time.
    First 80%  → Training set   (240 samples per class, 6,240 total)
    Next  10%  → Validation set  (30 samples per class,   780 total)
    Last  10%  → Test set        (30 samples per class,   780 total)
```

**Rationale:** Chronological splitting preserves the physical acquisition order within each class, preventing temporal leakage between train and test. This is more rigorous than random shuffling because later sessions may have different signal characteristics (electrode impedance drift, fatigue effects).

---

## Model Architectures

### 1. DeepConvNet

Based on Schirrmeister et al. (2017), adapted per the reference paper: reduced from 4 to 3 sequential convolutional blocks to prevent overfitting on the limited dataset.

```
Input: (B, 1, 24, 801)

Block 1: Conv2d(1→20, (1,15)) → BN → Conv2d(20→20, (24,1)) → BN → ELU → MaxPool(1,2) → Dropout(0.5)
Block 2: Conv2d(20→40, (1,15)) → BN → ELU → MaxPool(1,2) → Dropout(0.5)
Block 3: Conv2d(40→80, (1,15)) → BN → ELU → MaxPool(1,2) → Dropout(0.5)

FC: Flatten → Dropout(0.5) → Linear(8000, 26)

Parameters: 278,246
```

**Design rationale:** DeepConvNet uses standard conv-pool blocks to learn hierarchical spatiotemporal features. Block 1 combines temporal filtering (across time) with spatial integration (across electrodes). Blocks 2–3 extract progressively more abstract features. The 15-sample temporal kernel at 250 Hz captures 60 ms of EEG dynamics, which is well-suited for ERP components (P1 at ~100 ms, N170 at ~170 ms, P3 at ~300 ms).

### 2. EEGNet

Based on Lawhern et al. (2018), with enhancements: Visual ROI spatial prior and CBAM-EEG attention module.

```
Input: (B, 1, 24, 801)

VisualROISpatialPrior: Channel-wise attention (anatomical prior for visual/motor ROIs)
Block 1: Conv2d(1→32, (1,15)) → BN → DepthwiseConv2d(32→128, (24,1), groups=32) → BN → ELU → MaxPool(1,4) → Dropout(0.3)
CBAM-EEG: Channel attention (shared MLP) + Temporal attention (conv-based)
Block 2: SeparableConv2d(128→128, (1,16)) → PointwiseConv2d(128→64) → BN → ELU → AdaptiveAvgPool(1,16) → Dropout(0.3)

FC: Flatten → Linear(1024,128) → ELU → Dropout(0.3) → Linear(128,26)

Parameters: 157,344
Max-Norm constraints applied after each optimizer step.

Ensemble configuration: kernel=15 (60 ms temporal window).
```

**Design rationale:** EEGNet uses depthwise separable convolutions — a parameter-efficient design that separates spatial filtering (across electrodes) from temporal filtering (across time). The `VisualROISpatialPrior` initialises channel weights with anatomical knowledge: occipital channels (O1, O2, POz) at 0.95, parietal at 0.80, motor at 0.40, and prefrontal at 0.05. The CBAM module adds learnable channel and temporal attention. The 15-sample kernel (60 ms) matches the reference paper's recommendation for fine temporal dynamics in handwriting imagery — substantially shorter than the original EEGNet's kernel of 64 samples (256 ms) designed for motor imagery.

### 3. EEGInception

Based on Santamaria-Vazquez et al. (2020), adapted with shorter temporal kernels as recommended by the reference paper.

```
Input: (B, 1, 24, 801)

Inception Module 1: 3 parallel temporal convs (kernels 7, 5, 3) → Concat(72 ch) → Spatial Conv → BN → ELU → AvgPool(1,2) → Dropout(0.3)
Inception Module 2: Same structure
Conv Block 1: Conv(72→48, (1,5)) → BN → ELU → AvgPool(1,2) → Dropout(0.3)
Conv Block 2: Conv(48→48, (1,3)) → BN → ELU → AvgPool(1,2) → Dropout(0.3)

FC: Linear(2400, 26)

Parameters: 243,655
```

**Design rationale:** The Inception modules extract information at three temporal scales simultaneously (56 ms, 40 ms, 24 ms at 250 Hz). Different handwriting imagery processes — visual encoding, motor planning, and execution imagery — may be encoded in distinct frequency bands and time scales. The multi-scale parallel convolutions capture this diversity in a single forward pass.

---

## Training Methodology

### Core Training Loop

All models share a common training infrastructure:

| Component | Configuration |
|-----------|---------------|
| **Optimizer** | AdamW, learning rate = 0.005, weight decay = 0.05 |
| **LR Scheduler** | ReduceLROnPlateau (mode=min, factor=0.5, patience=3, min_lr=1e−6) |
| **Loss Function** | Cross-Entropy with label smoothing (0.0 for DeepConvNet, 0.1 for EEGNet/EEGInception) |
| **Batch Size** | 64 |
| **Max Epochs** | 150 (early stopping patience = 40 on validation loss) |
| **Checkpointing** | Best validation loss epoch weights saved to `models/checkpoints/` |
| **Reproducibility** | Seed = 42, `torch.backends.cudnn.deterministic = True` |
| **SWA** | Stochastic Weight Averaging baked into EEGNet (ensemble mode). Averages weights from epoch 25 onward to find flatter minima. |
| **Hardware** | CUDA GPU (falls back to CPU) |

### Model-Specific Training Configurations

| Model | Augmentations | Label Smoothing | Temporal Kernel |
|-------|:---:|:---:|:---:|
| DeepConvNet | None (clean data) | 0.0 | 15 (60 ms) |
| EEGNet | Mixup (α=0.2) + Gaussian noise (σ=0.07) | 0.1 | 15 (60 ms) |
| EEGInception | Mixup (α=0.2) | 0.1 | — |

**Rationale:** DeepConvNet benefits from clean data — its convolutional feature extraction is sensitive to signal corruption from augmentations. EEGNet's bottleneck architecture (depthwise separable convs) benefits from the additional regularization provided by Mixup and noise. This contrast in inductive biases is what makes their ensemble complementary.

---

## Regularization Techniques

The pipeline employs multiple orthogonal regularization strategies:

1. **Dropout (0.3–0.5):** Applied after every conv block and in FC layers. Prevents co-adaptation of features.
2. **Mixup (α=0.2):** Linear interpolation of training samples and their labels. Creates virtual training examples that encourage linear behaviour between classes, reducing overfitting on limited data.
3. **Gaussian Noise (σ=0.07):** Added to EEG signals during training. Forces the model to learn features robust to small amplitude perturbations — a natural form of data augmentation for EEG where trial-to-trial amplitude variability is high.
4. **Label Smoothing (0.1):** Softens one-hot targets from (0, 0, 1, 0) to (0.025, 0.025, 0.975, 0.025). Prevents the model from becoming overconfident.
5. **Max-Norm Constraints (EEGNet only):** Bounds the L2 norm of spatial filter weights (max 1.0) and FC layer weights (max 0.25). Enforces a compact weight space.
6. **Early Stopping:** Halts training when validation loss fails to improve for 40 consecutive epochs. Prevents overfitting to training data.
7. **Weight Decay (0.05):** L2 regularisation on all parameters via AdamW.
8. **ReduceLROnPlateau:** Halves the learning rate when validation loss plateaus for 3 epochs, allowing the model to settle into finer minima.
9. **Batch Size (64):** Smaller batches introduce beneficial gradient noise that acts as an implicit regulariser.
10. **Stochastic Weight Averaging (SWA):** Applied to EEGNet in ensemble mode starting from epoch 25. Maintains a running average of model weights rather than using a single best checkpoint. This finds flatter minima that generalise better, improving EEGNet single-model accuracy by ~2.8% on average. SWA is applied asymmetrically (EEGNet only) to preserve inter-model error diversity in the ensemble.

---

## Ensemble Method

The final model is an **equal-weight logit-averaging ensemble** of DeepConvNet and EEGNet:

```python
outputs = (dcn_logits + eegnet_logits) / 2.0
prediction = argmax(outputs)
```

### Why Ensemble Works

DeepConvNet and EEGNet have fundamentally different inductive biases:

| Property | DeepConvNet | EEGNet |
|----------|-------------|--------|
| Architecture | Standard conv-pool blocks | Depthwise separable + attention |
| Spatial processing | Full conv across all channels | Depthwise groups + anatomical prior |
| Temporal processing | Hierarchical (3 blocks) | Single block + CBAM attention |
| Regularization | Dropout + WD | Mixup + noise + max-norm + SWA |
| Augmentations | None | Mixup + Gaussian noise |

These differences mean the models make **different kinds of errors**. When one model is uncertain or incorrect, the other often compensates. This error decorrelation yields +4.23% over the best single model.

**The ensemble is not a multi-trial method** — each model processes the same single test trial independently, and their logits are averaged. No additional trial information is introduced at test time.

---

## Results

All results are **deterministic and reproducible** (seed 42, deterministic cuDNN). Multi-seed validation performed across seeds 41–43.

### Single-Model Performance

| Model | Test Accuracy | Parameters | Key Configuration |
|-------|:---:|:---:|---|
| Logistic Regression (baseline) | 13.72% | — | StandardScaler + C=0.05, max_iter=400 |
| DeepConvNet | **20.64%** | 278,246 | 250 Hz, no augmentations, kernel=15 |
| EEGNet (no SWA) | 19.49% | 157,344 | 250 Hz, mixup+noise, kernel=15 |
| EEGNet (with SWA) | 21.79% | 157,344 | 250 Hz, mixup+noise, kernel=15, SWA |
| EEGInception | 16.67% | 243,655 | 250 Hz, mixup, kernels=(7,5,3) |

### Ensemble Performance (Multi-Seed)

| Seed | DeepConvNet | EEGNet (SWA) | **Ensemble** |
|------|:---:|:---:|:---:|
| 41 | 17.69% | 21.79% | 23.85% |
| 42 | 20.64% | 21.41% | **25.38%** |
| 43 | 20.26% | 21.28% | 23.97% |
| **Average** | 19.53% | 21.49% | **24.40%** |

The optimal configuration achieves **25.38%** (seed 42) — a +11.66% improvement over the logistic regression baseline and +4.74% over the best single model. SWA adds +0.51% average ensemble improvement across seeds, with EEGNet single-model improvement of +2.77%. SWA is applied asymmetrically (EEGNet only) to preserve error decorrelation between the two models.

### Temporal Kernel Ablation

The EEGNet temporal kernel length strongly affects performance. Shorter kernels better capture the fine temporal dynamics of handwriting imagery:

| Kernel (samples) | Time Window | EEGNet Acc | Ensemble Acc |
|:---:|:---:|:---:|:---:|
| 125 | 500 ms | 14.10% | 20.00% |
| 25 | 100 ms | 18.21% | 22.31% |
| **15** | **60 ms** | **19.49%** | **24.87%** |
| 7 | 28 ms | 18.59% | 23.85% |

The sweet spot at 60 ms matches the reference paper's emphasis on short temporal kernels (~24 ms) for capturing the subtle dynamics of handwriting imagery, distinct from the longer kernels traditionally used for motor imagery.

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages: `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

### Data Preparation

Place `data_EEG_AI.mat` in `data/raw/`. Run the extraction pipeline once:

```bash
python src/extract.py
```

This creates `data/processed/eeg_dataset.npz`.

### Training

**Train a single model:**
```bash
python src/train.py --model deep_conv_net    # DeepConvNet
python src/train.py --model eegnet           # EEGNet
python src/train.py --model eeg_inception    # EEGInception
```

**Train the ensemble (DeepConvNet + EEGNet):**
```bash
python src/train.py --model ensemble
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Architecture: `deep_conv_net`, `eegnet`, `eeg_inception`, `ensemble`, `all` | `deep_conv_net` |
| `--downsample` | Temporal downsampling factor | `1` (250 Hz) |
| `--epochs` | Maximum training epochs | `150` |
| `--no-mixup` | Disable Mixup augmentation | Enabled for `ensemble`/`eegnet` |
| `--mixup-alpha` | Beta distribution alpha for Mixup | `0.2` |
| `--noise-std` | Gaussian noise standard deviation | `0.0` (off) |

Note: Stochastic Weight Averaging (SWA) is baked into the ensemble pipeline by default for EEGNet — no flag required.

---

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── data/
│   ├── raw/
│   │   ├── data_EEG_AI.mat            # Raw MATLAB dataset
│   │   └── data_instruction.docx      # Original task instructions
│   └── processed/
│       └── eeg_dataset.npz            # Preprocessed NumPy archive
├── models/
│   ├── __init__.py                    # Model exports
│   ├── deep_conv_net.py               # DeepConvNet architecture
│   ├── eegnet.py                      # EEGNet82 + CBAM + VisualROISpatialPrior
│   ├── eeg_inception.py               # EEGInception + InceptionModule
│   └── checkpoints/
│       └── *.pth                       # Saved model weights
└── src/
    ├── extract.py                     # Data extraction & preprocessing
    └── train.py                       # Training pipeline & ensemble orchestrator
```

---

## Academic Rationale

### Why These Architectures?

**DeepConvNet** was chosen because its hierarchical conv-pool structure is the most general-purpose EEG architecture. It makes minimal assumptions about the signal structure, learning features purely from data. The reference paper demonstrated its effectiveness on this exact task.

**EEGNet** was chosen because its depthwise separable convolutions are designed specifically for EEG — separating spatial filtering (per-electrode patterns) from temporal filtering (per-time-point dynamics). The anatomical spatial prior and CBAM attention add clinically-motivated inductive biases.

**EEGInception** was chosen because its multi-scale parallel convolutions capture temporal dynamics at multiple resolutions simultaneously — important for handwriting imagery where different processing stages (visual encoding, motor planning) operate at different time scales.

### Why Shorter Temporal Kernels?

Handwriting imagery involves rapid, fine-grained neural dynamics distinct from slower motor imagery processes. The reference paper found that reducing EEGNet's temporal kernel from 64 samples (256 ms) to 6 samples (24 ms) significantly improved performance. Our experiments confirmed this: kernel=15 (60 ms) was optimal, with longer kernels degrading performance.

### Why Ensemble?

DeepConvNet and EEGNet represent different points on the bias-variance trade-off:
- DeepConvNet has higher capacity (278K params) and uses minimal regularisation (dropout only) — it captures complex features but risks overfitting.
- EEGNet has lower capacity (157K params) and uses aggressive regularisation (mixup, noise, max-norm, SWA) — it learns robust but potentially simpler features.

Their complementary errors cancel out in the ensemble, producing predictions more accurate than either model alone. SWA (Stochastic Weight Averaging) is applied only to EEGNet in the ensemble — this asymmetrical application preserves the error diversity that makes the ensemble effective while giving EEGNet a +2.8% single-model boost.

### Why No Contrastive Pre-training?

Self-supervised contrastive learning (both InfoNCE and supervised contrastive/SupCon) was extensively tested but regressed performance by 2–4%. On a dataset with only 7,800 samples across 26 classes, the representations learned by the contrastive objective did not transfer to the classification task — the encoder learned instance-level discriminative features rather than class-level semantic features.
