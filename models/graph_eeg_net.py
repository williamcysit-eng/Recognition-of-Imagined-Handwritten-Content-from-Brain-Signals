import numpy as np
import torch
import torch.nn as nn


def electrode_graph(k=4):
    # Approximate 2-D 10-20 scalp positions in the dataset's channel order.
    coords = np.asarray([
        (-0.45, 1.00), (0.45, 1.00), (-0.50, 0.55), (0.50, 0.55),
        (-0.55, 0.00), (0.55, 0.00), (-0.50, -0.55), (0.50, -0.55),
        (-0.35, -1.00), (0.35, -1.00), (-0.95, 0.55), (0.95, 0.55),
        (-1.00, 0.00), (1.00, 0.00), (-0.95, -0.55), (0.95, -0.55),
        (0.00, 0.62), (0.00, 0.00), (0.00, -0.58), (-1.15, -0.05),
        (1.15, -0.05), (0.00, 0.90), (0.00, -0.30), (0.00, -0.88),
    ], dtype=np.float32)
    distance = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    adjacency = np.zeros_like(distance)
    for row in range(len(coords)):
        neighbours = np.argsort(distance[row])[1:k + 1]
        adjacency[row, neighbours] = np.exp(-distance[row, neighbours] ** 2 / 0.35)
    adjacency = np.maximum(adjacency, adjacency.T) + np.eye(len(coords), dtype=np.float32)
    degree = adjacency.sum(1)
    norm = np.diag(1.0 / np.sqrt(degree))
    return torch.from_numpy((norm @ adjacency @ norm).astype(np.float32))


class GraphTemporalBlock(nn.Module):
    def __init__(self, features, kernel, dropout):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1, features, 1, 1))
        self.temporal = nn.Sequential(
            nn.Conv2d(features, features, (1, kernel), padding=(0, kernel // 2),
                      groups=features, bias=False),
            nn.Conv2d(features, features, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, adjacency):
        graph_x = torch.einsum("ij,bfjt->bfit", adjacency, x)
        return x + self.temporal(x + torch.sigmoid(self.gate) * graph_x)


class GraphEEGNet(nn.Module):
    def __init__(self, num_channels=24, num_classes=26, dropout=0.35):
        super().__init__()
        self.register_buffer("adjacency", electrode_graph())
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.graph1 = GraphTemporalBlock(32, 9, dropout)
        self.graph2 = GraphTemporalBlock(32, 15, dropout)
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 64, (num_channels, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(64, 64, (1, 15), padding=(0, 7), groups=64, bias=False),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 16)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 16, 128), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.temporal(x)
        x = self.graph1(x, self.adjacency)
        x = self.graph2(x, self.adjacency)
        return self.head(self.spatial(x))


class DynamicGraphTemporalBlock(nn.Module):
    def __init__(self, features, kernel, dropout, graph_dim=16):
        super().__init__()
        self.query = nn.Linear(features, graph_dim, bias=False)
        self.key = nn.Linear(features, graph_dim, bias=False)
        self.mix = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Parameter(torch.zeros(1, features, 1, 1))
        self.scale = graph_dim ** -0.5
        self.temporal = nn.Sequential(
            nn.Conv2d(features, features, (1, kernel), padding=(0, kernel // 2),
                      groups=features, bias=False),
            nn.Conv2d(features, features, 1, bias=False),
            nn.BatchNorm2d(features), nn.ELU(), nn.Dropout(dropout),
        )

    def forward(self, x, physical):
        nodes = x.mean(dim=-1).transpose(1, 2)  # B, electrodes, features
        scores = torch.matmul(self.query(nodes), self.key(nodes).transpose(1, 2)) * self.scale
        dynamic = scores.softmax(dim=-1)
        blend = torch.sigmoid(self.mix)
        adjacency = (1 - blend) * physical.unsqueeze(0) + blend * dynamic
        graph_x = torch.einsum("bij,bfjt->bfit", adjacency, x)
        return x + self.temporal(x + torch.sigmoid(self.gate) * graph_x)


class DynamicGraphEEGNet(GraphEEGNet):
    """GraphEEGNet with trial-specific functional connectivity."""

    def __init__(self, num_channels=24, num_classes=26, dropout=0.35):
        super().__init__(num_channels, num_classes, dropout)
        self.graph1 = DynamicGraphTemporalBlock(32, 9, dropout)
        self.graph2 = DynamicGraphTemporalBlock(32, 15, dropout)


class MultiScaleGraphTemporalBlock(nn.Module):
    def __init__(self, features, kernel, dropout):
        super().__init__();self.graph_logits=nn.Parameter(torch.zeros(3));self.gate=nn.Parameter(torch.zeros(1,features,1,1))
        self.temporal=nn.Sequential(nn.Conv2d(features,features,(1,kernel),padding=(0,kernel//2),groups=features,bias=False),nn.Conv2d(features,features,1,bias=False),nn.BatchNorm2d(features),nn.ELU(),nn.Dropout(dropout))
    def forward(self,x,graphs):
        weights=self.graph_logits.softmax(0);mixed=sum(w*torch.einsum('ij,bfjt->bfit',g,x) for w,g in zip(weights,graphs))
        return x+self.temporal(x+torch.sigmoid(self.gate)*mixed)


class MultiScaleGraphEEGNet(GraphEEGNet):
    """GraphEEGNet using local, medium, and global fixed anatomical graphs."""
    def __init__(self,num_channels=24,num_classes=26,dropout=.35):
        super().__init__(num_channels,num_classes,dropout)
        self.register_buffer('multiscale_adjacency',torch.stack([electrode_graph(k) for k in (2,4,8)]))
        self.graph1=MultiScaleGraphTemporalBlock(32,9,dropout);self.graph2=MultiScaleGraphTemporalBlock(32,15,dropout)
    def forward(self,x):
        x=self.temporal(x);x=self.graph1(x,self.multiscale_adjacency);x=self.graph2(x,self.multiscale_adjacency)
        return self.head(self.spatial(x))
