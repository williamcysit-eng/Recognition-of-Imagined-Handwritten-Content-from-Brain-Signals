import torch
import torch.nn as nn

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
            nn.MaxPool2d(kernel_size=(1, 4))  # Downsamples sequence length
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