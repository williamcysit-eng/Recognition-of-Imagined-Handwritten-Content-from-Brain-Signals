import torch
import torch.nn as nn

class CBAM_EEG(nn.Module):
    """
    Temporal-Spectral (CBAM-EEG) Attention Block.
    Applies parallel Channel (spectral) Attention and Temporal Attention
    to dynamically prioritize informative electrodes/features and time points.
    """
    def __init__(self, channels, reduction=4, temporal_kernel_size=7):
        super(CBAM_EEG, self).__init__()
        
        # 1. Channel (Spectral) Attention Block
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for Channel Attention
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ELU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # 2. Temporal Attention Block
        # Since height is 1, we convolve along the temporal width (dimension 3)
        self.conv_temporal = nn.Conv2d(2, 1, kernel_size=(1, temporal_kernel_size), 
                                       padding=(0, temporal_kernel_size // 2), bias=False)
        self.sigmoid_temporal = nn.Sigmoid()
        
    def forward(self, x):
        # Input shape: (Batch, Channels, Height=1, Time_Points)
        
        # --- A. Channel Attention ---
        avg_out = self.mlp(self.channel_avg(x))
        max_out = self.mlp(self.channel_max(x))
        channel_weights = self.sigmoid_channel(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_weights
        
        # --- B. Temporal Attention ---
        # Pool along the channel dimension (dim=1)
        avg_temporal = torch.mean(x, dim=1, keepdim=True)
        max_temporal, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along the channel dimension to get a 2-channel spatial-temporal map
        temporal_features = torch.cat([avg_temporal, max_temporal], dim=1)
        temporal_weights = self.sigmoid_temporal(self.conv_temporal(temporal_features))
        x = x * temporal_weights
        
        return x


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


class EEGNet82(nn.Module):
    def __init__(self, num_channels=24, num_classes=26, F1=16, D=4, F2=64, 
                 input_time_points=401, temporal_kernel_length=64, dropout_rate=0.3):
        super(EEGNet82, self).__init__()
        
        self.num_channels = num_channels
        self.input_time_points = input_time_points
        
        # Anatomical Visual ROI Spatial Prior
        self.spatial_prior = VisualROISpatialPrior(num_channels=num_channels)
        
        # 1. Block 1: Temporal Conv -> Depthwise Spatial Conv
        self.temporal_conv = nn.Sequential(
            # Temporal convolution with long filter along time axis (1, kernel_length)
            nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel_length), 
                      padding=(0, temporal_kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # Spatial Convolution with depthwise group constraint (Standard EEGNet design)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 4)),  # Max pooling to capture transient signal peaks
            nn.Dropout(dropout_rate)
        )
        
        # Temporal-Spectral (CBAM-EEG) Attention Block
        self.cbam = CBAM_EEG(channels=F1 * D, reduction=4)
        
        # 2. Block 2: Separable Temporal Conv
        self.separable_conv = nn.Sequential(
            # Depthwise temporal convolution
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), 
                      groups=F1 * D, bias=False),
            # Pointwise convolution (mixes filters)
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)),  # Adaptive pooling for robust spatiotemporal representation
            nn.Dropout(dropout_rate)
        )
        
        # 3. Dense Classifier Head
        self.flat_features = self._get_fc_input_size()
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, 128),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def _get_fc_input_size(self):
        """
        Passes a dummy tensor through the convolutional layers to determine the correct 
        number of flattened features input to the Linear classifier.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.num_channels, self.input_time_points)
            dummy = self.spatial_prior(dummy)
            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = self.cbam(x)
            x = self.separable_conv(x)
            return x.numel()

    def forward(self, x):
        # x shape: (Batch, 1, channels, time_points)
        x = self.spatial_prior(x)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.cbam(x)
        x = self.separable_conv(x)
        x = self.fc(x)
        return x

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
        elif 'fc.4.weight' in name:
            norms = torch.norm(param.data, p=2, dim=1, keepdim=True)
            desired = torch.clamp(norms, max=max_norm_fc)
            param.data *= (desired / (norms + 1e-10))



