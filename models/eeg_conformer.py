import torch
import torch.nn as nn


class EEGConformer(nn.Module):
    """Compact convolutional Transformer for time-locked EEG decoding."""

    def __init__(self, num_channels=24, num_classes=26, time_points=501,
                 embed_dim=64, depth=3, heads=4, dropout=0.35):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12), bias=False),
            nn.Conv2d(40, 40, (num_channels, 1), bias=False),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 25), stride=(1, 10)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = self.tokenizer(torch.zeros(1, 1, num_channels, time_points))
            token_count = dummy.shape[-1]
        self.projection = nn.Linear(40, embed_dim)
        self.position = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth,
                                             norm=nn.LayerNorm(embed_dim))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(token_count * embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x):
        x = self.tokenizer(x).squeeze(2).transpose(1, 2)
        x = self.projection(x) + self.position
        return self.head(self.encoder(x))
