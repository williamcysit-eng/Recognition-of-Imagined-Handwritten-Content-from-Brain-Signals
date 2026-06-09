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


class EEGConformer(nn.Module):
    """
    Optimized EEG-Conformer Hybrid Model
    Parameter Footprint: ~85,000 Parameters
    """
    def __init__(self, num_channels=24, num_classes=26, temporal_filters=40, 
                 num_heads=4, num_transformer_layers=2, forward_expansion=4, 
                 dropout_rate=0.3, input_time_points=81):
        super(EEGConformer, self).__init__()
        
        # 1. Local Spatiotemporal Feature Extraction (CNN Block)
        self.conv = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 15), stride=1, padding=(0, 7), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            # Grouped depthwise spatial convolution over electrodes (Standard EEG design)
            nn.Conv2d(temporal_filters, temporal_filters, kernel_size=(num_channels, 1), groups=temporal_filters, bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 4)),  # Downsamples sequence length
            CBAM_EEG(channels=temporal_filters, reduction=4)
        )
        
        # Robust Dynamic Shape Calculation using a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_channels, input_time_points)
            dummy_output = self.conv(dummy_input)
            self.seq_len = dummy_output.shape[3]
            self.feature_dim = dummy_output.shape[1]
        
        # 2. Learnable Positional Encoding (Fixes the permutation-invariance flaw)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.feature_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        
        # 3. Transformer Self-Attention Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            dim_feedforward=self.feature_dim * forward_expansion,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 4. Lightweight Classifier Head (Fixes the 102k parameter explosion)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: (Batch, 1, Channels, Time)
        x = self.conv(x)        # (Batch, Filters, 1, Seq_Len)
        x = x.squeeze(2)        # (Batch, Filters, Seq_Len)
        x = x.permute(0, 2, 1)  # (Batch, Seq_Len, Feature_Dim)
        
        # Inject chronological temporal awareness
        x = x + self.pos_embedding
        x = self.pos_drop(x)
        
        # Contextualize long-range dependencies
        x = self.transformer(x) # (Batch, Seq_Len, Feature_Dim)
        
        # Global Average Pooling along the time dimension
        x = x.mean(dim=1)       # (Batch, Feature_Dim)
        
        return self.fc(x)