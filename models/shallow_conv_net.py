import torch
import torch.nn as nn


class Square(nn.Module):
    def forward(self, x):
        return x.square()


class SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(x.clamp_min(1e-6))


class ShallowConvNet(nn.Module):
    """FBCSP-inspired network from the standard EEG decoding literature."""

    def __init__(self, num_channels=24, num_classes=26, time_points=801,
                 temporal_filters=40, spatial_filters=40, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, temporal_filters, (1, 25), bias=False),
            nn.Conv2d(temporal_filters, spatial_filters, (num_channels, 1), bias=False),
            nn.BatchNorm2d(spatial_filters),
            Square(),
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            SafeLog(),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_features = self.features(torch.zeros(1, 1, num_channels, time_points)).numel()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(n_features, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))
