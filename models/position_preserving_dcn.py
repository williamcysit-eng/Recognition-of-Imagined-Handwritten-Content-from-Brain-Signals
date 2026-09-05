"""DeepConvNet variant with a smaller position-preserving classifier head."""

from __future__ import annotations

import torch
from torch import nn

from .deep_conv_net import DeepConvNet


class PositionPreservingDeepConvNet(DeepConvNet):
    """DeepConvNet with a shared 1x1 feature bottleneck.

    The convolutional stem is unchanged. Its 80 feature channels are projected
    to 32 channels at every output time position, then all 100 positions are
    flattened for classification. No temporal pooling is added to the head.
    """

    def __init__(
        self,
        num_channels: int = 24,
        num_classes: int = 26,
        input_time_points: int = 801,
        F1: int = 20,
        F2: int = 40,
        F3: int = 80,
        temporal_kernel: int = 15,
        dropout_rate: float = 0.5,
        projection_features: int = 32,
    ) -> None:
        if projection_features <= 0:
            raise ValueError("projection_features must be positive")
        super().__init__(
            num_channels=num_channels,
            num_classes=num_classes,
            input_time_points=input_time_points,
            F1=F1,
            F2=F2,
            F3=F3,
            temporal_kernel=temporal_kernel,
            dropout_rate=dropout_rate,
        )
        self.feature_channels = int(F3)
        self.projection_features = int(projection_features)
        self.feature_projection = nn.Conv2d(
            self.feature_channels,
            self.projection_features,
            kernel_size=(1, 1),
            bias=False,
        )
        output_time_positions = self._flat_features // self.feature_channels
        self.output_time_positions = int(output_time_positions)
        self._flat_features = self.projection_features * self.output_time_positions
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self._flat_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.feature_projection(x)
        return self.fc(x)
