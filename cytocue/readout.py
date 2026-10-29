import torch
from torch import nn
from torch.nn import functional as F


class ContextualGatedReadout(nn.Module):
    def __init__(self, dimension=1536, bottleneck=64, width=160):
        super().__init__()
        self.preprocessing = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, bottleneck), nn.Tanh()
        )
        self.gate = nn.Linear(bottleneck, width)
        self.value = nn.Linear(bottleneck, width)
        self.output = nn.Linear(width, 1)

    def forward(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden = self.preprocessing(features.float())
            return self.output(F.silu(self.gate(hidden)) * self.value(hidden)).squeeze(-1)
