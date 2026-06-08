import torch
import torch.nn as nn

class EEGNet82(nn.Module):
    """
    Standard EEGNet architecture as proposed in:
    "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces"
    (https://iopscience.iop.org/article/10.1088/1741-2552/aae5d2)
    
    Parameters:
    -----------
    num_channels: int
        Number of EEG electrodes (default: 24).
    num_classes: int
        Number of target classes (default: 26).
    F1: int
        Number of temporal filters (default: 8).
    D: int
        Depth multiplier for spatial filters (default: 2, yielding F2 = F1 * D = 16).
    F2: int
        Number of pointwise filters (default: 16).
    input_time_points: int
        Length of the time sequence (default: 401 for downsample_factor=2).
    temporal_kernel_length: int
        Size of temporal filters, scaled to sampling rate. Typically sampling_rate // 2.
    """
    def __init__(self, num_channels=24, num_classes=26, F1=8, D=2, F2=16, 
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
        
        # Depthwise Spatial Convolution (constrained to groups=F1 to isolate filters per channel)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # EEGNet uses average pooling
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
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # 3. Dense Classifier Head
        self.flat_features = self._get_fc_input_size()
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, num_classes)
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


class EEGNetEncoder(nn.Module):
    """
    EEGNet feature extractor that produces flat spatiotemporal embedding vectors.
    """
    def __init__(self, num_channels=24, input_time_points=401, temporal_kernel_length=64,
                 F1=8, D=2, F2=16, dropout_rate=0.3):
        super(EEGNetEncoder, self).__init__()
        self.base = EEGNet82(
            num_channels=num_channels,
            num_classes=2,  # Dummy value for instantiating the base EEGNet FC size
            F1=F1, D=D, F2=F2,
            input_time_points=input_time_points,
            temporal_kernel_length=temporal_kernel_length,
            dropout_rate=dropout_rate
        )
        self.latent_dim = self.base.flat_features

    def forward(self, x):
        x = self.base.temporal_conv(x)
        x = self.base.spatial_conv(x)
        x = self.base.separable_conv(x)
        x = torch.flatten(x, start_dim=1)
        return x


class EEGSpatiotemporalMasker(nn.Module):
    """
    Applies spatiotemporal masking to raw EEG signals.
    Randomly zeros out continuous blocks of time across specific channel clusters.
    """
    def __init__(self, mask_ratio_time=0.3, mask_ratio_channels=0.3, min_block_len=10):
        super(EEGSpatiotemporalMasker, self).__init__()
        self.mask_ratio_time = mask_ratio_time
        self.mask_ratio_channels = mask_ratio_channels
        self.min_block_len = min_block_len

    def forward(self, x):
        if not self.training:
            return x
            
        device = x.device
        batch_size, _, num_channels, time_samples = x.shape
        masked_x = x.clone()
        
        for b in range(batch_size):
            num_masked_chans = max(1, int(num_channels * self.mask_ratio_channels))
            masked_chans = torch.randperm(num_channels, device=device)[:num_masked_chans]
            
            block_len = max(self.min_block_len, int(time_samples * self.mask_ratio_time))
            if time_samples > block_len:
                t_start = torch.randint(0, time_samples - block_len, (1,), device=device).item()
                t_end = t_start + block_len
                masked_x[b, 0, masked_chans, t_start:t_end] = 0.0
                
        return masked_x


class EEGJEPAPredictor(nn.Module):
    """
    Predicts the target latent embedding from the masked context embedding.
    """
    def __init__(self, latent_dim, hidden_dim=256):
        super(EEGJEPAPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.predictor(x)


class EEGJEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture (JEPA) for EEG spatiotemporal representation learning.
    """
    def __init__(self, num_channels=24, input_time_points=401, temporal_kernel_length=64, 
                 F1=8, D=2, F2=16, dropout_rate=0.3, mask_ratio_time=0.3, 
                 mask_ratio_channels=0.3, min_block_len=10, ema_decay=0.99):
        super(EEGJEPA, self).__init__()
        
        self.ema_decay = ema_decay
        
        self.online_encoder = EEGNetEncoder(
            num_channels=num_channels,
            input_time_points=input_time_points,
            temporal_kernel_length=temporal_kernel_length,
            F1=F1, D=D, F2=F2,
            dropout_rate=dropout_rate
        )
        
        self.target_encoder = EEGNetEncoder(
            num_channels=num_channels,
            input_time_points=input_time_points,
            temporal_kernel_length=temporal_kernel_length,
            F1=F1, D=D, F2=F2,
            dropout_rate=dropout_rate
        )
        
        self.reset_target_encoder()
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.masker = EEGSpatiotemporalMasker(
            mask_ratio_time=mask_ratio_time,
            mask_ratio_channels=mask_ratio_channels,
            min_block_len=min_block_len
        )
        
        self.predictor = EEGJEPAPredictor(
            latent_dim=self.online_encoder.latent_dim,
            hidden_dim=256
        )

    def reset_target_encoder(self):
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())

    @torch.no_grad()
    def update_target_ema(self):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.mul_(self.ema_decay).add_(online_param.data, alpha=1.0 - self.ema_decay)

    def forward(self, raw_eeg):
        with torch.no_grad():
            target_embeddings = self.target_encoder(raw_eeg)
            
        masked_eeg = self.masker(raw_eeg)
        context_embeddings = self.online_encoder(masked_eeg)
        predicted_embeddings = self.predictor(context_embeddings)
        
        return predicted_embeddings, target_embeddings


def compute_sigreg_loss(embeddings):
    """
    SIGReg (Gaussian Regularization): Enforces that the dimensions of the latent 
    embeddings are Gaussian-distributed.
    This is achieved by penalizing deviations of the mean from 0 and the standard
    deviation from 1.0 across the batch.
    """
    # Prevent divide by zero / NaN in case of standard deviation on small batch sizes
    eps = 1e-6
    
    # Mean of each feature across the batch should be 0
    mean_loss = torch.mean(embeddings.mean(dim=0) ** 2)
    
    # Standard deviation of each feature across the batch should be 1.0
    # Add a small epsilon to standard deviation computation for numerical stability
    std_dev = torch.sqrt(embeddings.var(dim=0, unbiased=False) + eps)
    std_loss = torch.mean((std_dev - 1.0) ** 2)
    
    return mean_loss + std_loss


class EEGJEPAClassifier(nn.Module):
    """
    Downstream classification model that wraps a pre-trained EEGNetEncoder.
    Can be used for Linear Probing (encoder frozen) or Fine-Tuning (encoder trainable).
    """
    def __init__(self, encoder, num_classes=26, freeze_encoder=True):
        super(EEGJEPAClassifier, self).__init__()
        self.encoder = encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # A 2-layer projection classification head similar to LightEEG2DCNN's fc to keep capacity high
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Extract embeddings
        z = self.encoder(x)
        # Classify
        return self.fc(z)
