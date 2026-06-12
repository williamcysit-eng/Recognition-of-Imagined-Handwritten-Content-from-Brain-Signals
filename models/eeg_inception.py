import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_per_branch, num_channels, kernels=(7, 5, 3),
                 pool_size=(1, 2), dropout_rate=0.3):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, out_per_branch, kernel_size=(1, kernels[0]),
                                  padding=(0, kernels[0] // 2), bias=False)
        self.branch2 = nn.Conv2d(in_channels, out_per_branch, kernel_size=(1, kernels[1]),
                                  padding=(0, kernels[1] // 2), bias=False)
        self.branch3 = nn.Conv2d(in_channels, out_per_branch, kernel_size=(1, kernels[2]),
                                  padding=(0, kernels[2] // 2), bias=False)

        total_out = out_per_branch * 3
        self.spatial_conv = nn.Conv2d(total_out, total_out, kernel_size=(num_channels, 1),
                                       bias=False)
        self.bn = nn.BatchNorm2d(total_out)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        x = torch.cat([b1, b2, b3], dim=1)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class EEGInception(nn.Module):
    def __init__(self, num_channels=24, num_classes=26, input_time_points=401,
                 F_per_branch=24, dropout_rate=0.3):
        super(EEGInception, self).__init__()

        self.inception1 = InceptionModule(
            in_channels=1, out_per_branch=F_per_branch,
            num_channels=num_channels, kernels=(7, 5, 3),
            pool_size=(1, 2), dropout_rate=dropout_rate
        )

        self.inception2 = InceptionModule(
            in_channels=F_per_branch * 3, out_per_branch=F_per_branch,
            num_channels=1, kernels=(7, 5, 3),
            pool_size=(1, 2), dropout_rate=dropout_rate
        )

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(F_per_branch * 3, F_per_branch * 2, kernel_size=(1, 5),
                      padding=(0, 2), bias=False),
            nn.BatchNorm2d(F_per_branch * 2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(F_per_branch * 2, F_per_branch * 2, kernel_size=(1, 3),
                      padding=(0, 1), bias=False),
            nn.BatchNorm2d(F_per_branch * 2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate),
        )

        self._flat_features = self._compute_flat_size(num_channels, input_time_points)
        self.fc = nn.Linear(self._flat_features, num_classes)

    def _compute_flat_size(self, num_channels, input_time_points):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, input_time_points)
            x = self.inception1(dummy)
            x = self.inception2(x)
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            return x.numel()

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
