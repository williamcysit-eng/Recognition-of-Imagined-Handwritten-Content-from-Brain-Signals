import torch
import torch.nn as nn


class DeepConvNet(nn.Module):
    """
    DeepConvNet with depthwise spatial convolution in Block 1
    (EEGNet-inspired) for better feature separation per temporal filter.
    """
    def __init__(self, num_channels=24, num_classes=26, input_time_points=401,
                 F1=20, F2=40, F3=80, temporal_kernel=5, dropout_rate=0.5):
        super(DeepConvNet, self).__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        # Depthwise spatial: separate spatial filter per temporal feature
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.Conv2d(F1, F1, kernel_size=1, bias=False),  # Pointwise mix
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(F2, F3, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(F3),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate),
        )

        self._flat_features = self._compute_flat_size(num_channels, input_time_points)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self._flat_features, num_classes),
        )

    def _compute_flat_size(self, num_channels, input_time_points):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, input_time_points)
            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = self.block2(x)
            x = self.block3(x)
            return x.numel()

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)
        return x
