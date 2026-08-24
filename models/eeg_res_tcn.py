import torch
import torch.nn as nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3 * dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 7, padding=3 * dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class EEGResTCN(nn.Module):
    """Multi-scale residual temporal network for raw EEG trials."""

    def __init__(self, num_channels=24, num_classes=26, width=64, dropout=0.35):
        super().__init__()
        branch_width = width // 4
        kernels = (7, 15, 31, 63)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_channels, branch_width, kernel, padding=kernel // 2,
                          bias=False),
                nn.GroupNorm(4, branch_width),
                nn.GELU(),
            ) for kernel in kernels
        ])
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1, bias=False),
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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 16, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.head_pool = nn.AdaptiveAvgPool1d(16)

    def forward(self, x):
        x = x.squeeze(1)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.temporal(self.project(x))
        return self.head(self.head_pool(x))
