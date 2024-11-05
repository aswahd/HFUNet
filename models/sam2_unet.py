
from typing import List

import torch
import torch.nn as nn

from SAM2UNet.SAM2UNet import SAM2UNet
from utils import DotDict

from .model_registry import register_model


@register_model("sam2_unet")
class SAM2UNetModel(nn.Module):
    """
    Sam2UNet model: https://www.arxiv.org/abs/2408.08870
    """

    def __init__(self, config: DotDict):
        super().__init__()

        self.model = SAM2UNet(checkpoint_path=config.checkpoint)

        self.num_classes = config.dataset.num_classes

    def forward(self, x) -> List[torch.Tensor]:
        return self.model(x)
