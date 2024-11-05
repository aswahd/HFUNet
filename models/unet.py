from typing import List

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from utils import DotDict

from .model_registry import register_model


@register_model("unet")
class UNet(nn.Module):
    def __init__(self, config: DotDict):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=config.encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config.encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=config.in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=config.dataset.num_classes,  # model output channels (number of classes in your dataset)
        )

        self.num_classes = config.dataset.num_classes

    def forward(self, x) -> List[torch.Tensor]:
        outputs = self.model(x)
        if not isinstance(outputs, list):
            return [outputs]

        return outputs
