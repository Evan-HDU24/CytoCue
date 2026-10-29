import torch
from torch import nn
from .encoding import CellNeighborhoodEncoding
from .readout import ContextualGatedReadout


class CytoCue(nn.Module):
    def __init__(self, dimension=512, bottleneck=64, width=160, pretrained=False):
        super().__init__()
        self.encoder = CellNeighborhoodEncoding(dimension, pretrained)
        self.readout = ContextualGatedReadout(self.encoder.output_dimension, bottleneck, width)

    def forward(self, batch):
        return self.readout(self.encoder(batch))


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CytoCue(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device, memory_format=torch.channels_last).eval()
