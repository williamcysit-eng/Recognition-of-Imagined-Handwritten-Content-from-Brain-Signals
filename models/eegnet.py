import torch
import torch.nn as nn

class EEGNet82(nn.Module):
    def __init__(self, num_channels=24, num_classes=26, F1=16, D=4, F2=64, 
                 input_time_points=401, temporal_kernel_length=64, dropout_rate=0.3):
        super(EEGNet82, self).__init__()
        
        self.num_channels = num_channels
        self.input_time_points = input_time_points
        
        # 1. Block 1: Temporal Conv -> Depthwise Spatial Conv
        self.temporal_conv = nn.Sequential(
            # Temporal convolution with long filter along time axis (1, kernel_length)
            nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel_length), 
                      padding=(0, temporal_kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # Spatial Convolution with depthwise group constraint (Standard EEGNet design)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 4)),  # Max pooling to capture transient signal peaks
            nn.Dropout(dropout_rate)
        )
        
        # 2. Block 2: Separable Temporal Conv
        self.separable_conv = nn.Sequential(
            # Depthwise temporal convolution
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), 
                      groups=F1 * D, bias=False),
            # Pointwise convolution (mixes filters)
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)),  # Adaptive pooling for robust spatiotemporal representation
            nn.Dropout(dropout_rate)
        )
        
        # 3. Dense Classifier Head
        self.flat_features = self._get_fc_input_size()
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, 128),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def _get_fc_input_size(self):
        """
        Passes a dummy tensor through the convolutional layers to determine the correct 
        number of flattened features input to the Linear classifier.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.num_channels, self.input_time_points)
            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
            return x.numel()

    def forward(self, x):
        # x shape: (Batch, 1, channels, time_points)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = self.fc(x)
        return x

@torch.no_grad()
def apply_max_norm_constraints(model, max_norm_spatial=1.0, max_norm_fc=0.25):
    """
    Applies Max-Norm weight constraints after the optimizer step as described in the 
    original EEGNet publication. This bounds individual filters to prevent overfitting.
    """
    for name, param in model.named_parameters():
        # Constrain spatial depthwise weights across channels (dim=2)
        if 'spatial_conv.0.weight' in name:
            norms = torch.norm(param.data, p=2, dim=2, keepdim=True)
            desired = torch.clamp(norms, max=max_norm_spatial)
            param.data *= (desired / (norms + 1e-10))
            
        # Constrain Dense Classification layer weights (dim=1)
        elif 'fc.1.weight' in name:
            norms = torch.norm(param.data, p=2, dim=1, keepdim=True)
            desired = torch.clamp(norms, max=max_norm_fc)
            param.data *= (desired / (norms + 1e-10))



