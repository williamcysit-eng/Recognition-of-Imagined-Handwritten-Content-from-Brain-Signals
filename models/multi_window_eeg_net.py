import torch
import torch.nn as nn


class WindowBranch(nn.Module):
    def __init__(self, channels=24, filters=16, dropout=0.35):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, filters, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters * 2, (channels, 1), groups=filters, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(filters * 2, filters * 2, (1, 9), padding=(0, 4),
                      groups=filters * 2, bias=False),
            nn.Conv2d(filters * 2, filters * 2, 1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)),
        )

    def forward(self, x):
        return self.features(x).flatten(1)


class MultiWindowEEGNet(nn.Module):
    """Separate branches for early, middle, and late post-stimulus EEG."""

    # Indices relative to the complete -200..3000 ms input.
    windows = ((50, 126), (125, 301), (300, 551))

    def __init__(self, num_classes=26, dropout=0.35):
        super().__init__()
        self.branches = nn.ModuleList([WindowBranch(dropout=dropout) for _ in self.windows])
        branch_features = 32 * 8
        self.head = nn.Sequential(
            nn.Linear(branch_features * len(self.windows), 256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = [branch(x[..., start:stop])
                    for branch, (start, stop) in zip(self.branches, self.windows)]
        return self.head(torch.cat(features, dim=1))
