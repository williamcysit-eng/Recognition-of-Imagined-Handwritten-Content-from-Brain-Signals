import torch
import torch.nn as nn

class LightEEG2DCNN(nn.Module):
    """
    Upgraded Hybrid Spatiotemporal CNN-GRU model for EEG classification.
    Uses 2D CNN layers to extract spatial-temporal features, and a Bidirectional 
    GRU to model sequential handwriting trajectory patterns over time.
    """
    def __init__(self, num_classes=26):
        super(LightEEG2DCNN, self).__init__()
        
        # 1. Temporal feature extraction (increased filter size and capacity)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 15), stride=1, padding=(0, 7), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d((1, 2)), # Half the time dimension
        )
        
        # 2. Spatial electrode feature mixing (increased capacity)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(24, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((1, 2)), # Quarter the time dimension
        )
        
        # 3. Dynamic temporal sequential modeling via Bidirectional GRU
        # Input shape to GRU: (Batch, Sequence_Length, 32)
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # 4. Dense Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 1, 24, T)
        x = self.temporal_conv(x)  # (Batch, 16, 24, T // 2)
        x = self.spatial_conv(x)   # (Batch, 32, 1, T // 4)
        
        # Reshape for GRU: (Batch, Channels, Height, Width) -> (Batch, Width, Channels)
        x = x.squeeze(2)          # (Batch, 32, T // 4)
        x = x.permute(0, 2, 1)    # (Batch, T // 4, 32)
        
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (Batch, T // 4, hidden_size * 2)
        
        # Pool across temporal sequence (average pooling over sequence dimension)
        feat = torch.mean(gru_out, dim=1)  # (Batch, 128)
        
        return self.fc(feat)
