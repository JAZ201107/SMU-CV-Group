import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


class vit(nn.Module):
    def __init__(self, params):
        super().__init__()
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_feature = model.heads.head.in_features
        model.heads = nn.Sequential(
            nn.Linear(in_feature, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=False),
        )
        self.model = model

    def forward(self, x):
        x = self.model(x)

        return torch.sigmoid(x)  # make range (0, 1)
