import torch
import torch.nn as nn

class VisualROISpatialPrior(nn.Module):
    """
    Anatomical Visual ROI Spatial Prior (Visual-Occipital Attention).
    Applies learnable attention weights over electrodes initialized with 
    strong priors for visual-evoked potentials (occipital/parietal channels)
    and suppresses frontal ocular noise channels.
    """
    def __init__(self, num_channels=24):
        super(VisualROISpatialPrior, self).__init__()
        # Channel names matching the dataset's order
        channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'M1', 
                    'M2', 'AFz', 'CPz', 'POz']
        
        # High priors for occipital/parietal (visual cortex), low for prefrontal (ocular noise)
        priors = []
        for ch in channels:
            ch_lower = ch.lower()
            if ch_lower in ['o1', 'o2', 'poz']:
                priors.append(0.95)  # Occipital / visual primary cortex
            elif ch_lower in ['p3', 'p4', 'pz', 'p7', 'p8', 'cpz']:
                priors.append(0.80)  # Parietal / visual word form area
            elif ch_lower in ['c3', 'c4', 'cz']:
                priors.append(0.40)  # Central / motor cortex
            elif ch_lower in ['f3', 'f4', 'fz', 'f7', 'f8', 't7', 't8', 'm1', 'm2']:
                priors.append(0.20)  # Frontal & temporal background channels
            else:
                priors.append(0.05)  # Prefrontal / AFz / ocular blink channels
                
        # Convert priors to initial logit weights
        init_logits = torch.log(torch.tensor(priors) / (1.0 - torch.tensor(priors) + 1e-5))
        self.weights = nn.Parameter(init_logits)
        
    def forward(self, x):
        # x shape: (Batch, 1, Channels, Time)
        # Apply sigmoid to weights to scale attention between 0.0 and 1.0
        attn = torch.sigmoid(self.weights).view(1, 1, -1, 1)
        return x * attn


class LightEEG2DCNN(nn.Module):
    """
    A highly optimized, ultra-lightweight 2D CNN for EEG classification.
    Designed for fast CPU training while learning powerful spatial-temporal features.
    """
    def __init__(self, num_classes=26):
        super(LightEEG2DCNN, self).__init__()
        
        # Anatomical Visual ROI Spatial Prior
        self.spatial_prior = VisualROISpatialPrior(num_channels=24)
        
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
        x = self.spatial_prior(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
