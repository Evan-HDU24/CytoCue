import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, Swin_T_Weights, resnet18, swin_t


def masked_pool(features, mask):
    weights = F.interpolate(mask.float(), size=features.shape[-2:], mode="area").clamp(0, 1)
    return (features * weights).sum((-2, -1)) / weights.sum((-2, -1)).clamp_min(1e-6)


class CellNeighborhoodEncoding(nn.Module):
    def __init__(self, dimension=512, pretrained=False):
        super().__init__()
        self.cell_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cell_encoder.fc = nn.Identity()
        context = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        self.context_encoder = context.features[:6]
        self.context_projection = nn.Linear(384, dimension)
        self.output_dimension = 512 + 2 * dimension

    def forward(self, batch):
        body = self.cell_encoder
        cell = body.maxpool(body.relu(body.bn1(body.conv1(batch["fine"]))))
        cell = body.layer4(body.layer3(body.layer2(body.layer1(cell))))
        context = self.context_encoder(batch["context"]).permute(0, 3, 1, 2).contiguous()
        return torch.cat(
            [
                masked_pool(cell, batch["fine_target"]),
                self.context_projection(masked_pool(context, batch["context_target"])),
                self.context_projection(masked_pool(context, batch["context_neighbor"])),
            ],
            dim=1,
        )

    def freeze_backbones(self, frozen):
        for module in (self.cell_encoder, self.context_encoder):
            module.requires_grad_(not frozen)
            module.train(not frozen)
