import torch
import torch.nn as nn


class DeepConvNet(nn.Module):
    """
    DeepConvNet adapted for handwriting imagery EEG classification.
    Per the reference paper: reduced from 4 to 2 sequential conv blocks
    to prevent overfitting on the limited dataset (300 samples/class).
    
    Architecture adapted from Schirrmeister et al. (2017):
    Block 1: Temporal conv + Spatial conv → BN → ELU → MaxPool → Dropout
    Block 2: Standard conv block → BN → ELU → MaxPool → Dropout
    FC classifier head.
    """
    def __init__(self, num_channels=24, num_classes=26, input_time_points=401,
                 F1=25, F2=50, temporal_kernel=5, dropout_rate=0.5):
        super(DeepConvNet, self).__init__()

        # Block 1: Temporal convolution (extracts dynamics across channels)
        # followed by spatial filter (integrates information across channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.Conv2d(F1, F1, kernel_size=(num_channels, 1), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: Standard convolutional block (extracts deeper features)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout_rate),
        )

        self._flat_features = self._compute_flat_size(num_channels, input_time_points)
        self.fc = nn.Linear(self._flat_features, num_classes)

    def _compute_flat_size(self, num_channels, input_time_points):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, input_time_points)
            x = self.block1(dummy)
            x = self.block2(x)
            return x.numel()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
