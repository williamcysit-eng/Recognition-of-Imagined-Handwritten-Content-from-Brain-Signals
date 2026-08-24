import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin

from .eeg_res_tcn import ResidualTemporalBlock


class FilterBankNet(nn.Module):
    """EEG classifier with a fixed physiological FIR filter bank."""

    def __init__(self, num_channels=24, num_classes=26, sample_rate=250,
                 taps=63, width=96, dropout=0.4):
        super().__init__()
        bands = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
        filters = np.stack([
            firwin(taps, band, pass_zero=False, fs=sample_rate).astype(np.float32)
            for band in bands
        ])
        # Grouped convolution expects all five filters for channel 0, then channel 1, etc.
        filters = np.tile(filters[:, None, :], (num_channels, 1, 1))
        self.register_buffer("filters", torch.from_numpy(filters))
        self.num_channels = num_channels
        self.padding = taps // 2

        self.fusion = nn.Sequential(
            nn.Conv1d(num_channels * len(bands), width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.AvgPool1d(4),
        )
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(width, 1, dropout),
            nn.AvgPool1d(2),
            ResidualTemporalBlock(width, 2, dropout),
            nn.AvgPool1d(2),
            ResidualTemporalBlock(width, 4, dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 16, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = F.conv1d(x, self.filters, padding=self.padding, groups=self.num_channels)
        x = self.temporal(self.fusion(x))
        return self.head(self.pool(x))
