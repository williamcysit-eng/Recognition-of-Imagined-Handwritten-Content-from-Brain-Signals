import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(24,64,15,stride=2,padding=7,bias=False),nn.BatchNorm1d(64),nn.GELU(),
            nn.Conv1d(64,96,9,stride=2,padding=4,bias=False),nn.BatchNorm1d(96),nn.GELU(),
            nn.Conv1d(96,128,7,stride=2,padding=3,bias=False),nn.BatchNorm1d(128),nn.GELU())
    def forward(self,x):return self.net(x.squeeze(1))


class MaskedEEGAutoencoder(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=EEGEncoder()
        self.decoder=nn.Sequential(nn.Conv1d(128,64,9,padding=4),nn.GELU(),nn.Conv1d(64,24,9,padding=4))
    def forward(self,x):
        z=self.encoder(x);z=F.interpolate(z,size=x.size(-1),mode='linear',align_corners=False)
        return self.decoder(z).unsqueeze(1)


class PretrainedEEGClassifier(nn.Module):
    def __init__(self,num_classes=26,dropout=.4):
        super().__init__();self.encoder=EEGEncoder();self.pool=nn.AdaptiveAvgPool1d(16)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128*16,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,num_classes))
    def forward(self,x):return self.head(self.pool(self.encoder(x)))
