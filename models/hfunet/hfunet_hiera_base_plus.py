import torch.nn as nn
from utils import DotDict

from ..decoders.registry import MASK_DECODER_REGISTRY
from ..encoders.registry import IMAGE_ENCODER_REGISTRY
from ..model_registry import register_model


@register_model("hfunet_hiera_base_plus")
class HFUNet(nn.Module):
    """
    UNet architecture with a Tiny ViT encoder and a transformer decoder.
    """

    def __init__(self, config: DotDict):
        super().__init__()

        config.image_encoder = DotDict.from_dict(
            {
                "name": "hiera_base_plus_blockwise_adapter",
                "embed_dims": [256, 256, 256, 256],
                "strides": [2, 2, 2],
            }
        )
        config.decoder = DotDict.from_dict(
            {
                "name": "mask_decoder",
                "num_classes": config.dataset.num_classes,
            }
        )

        self.encoder = IMAGE_ENCODER_REGISTRY[config.image_encoder.name].build(config)
        self.encoder.load_pretrained_weights(config.checkpoint)

        self.decoder = MASK_DECODER_REGISTRY[config.decoder.name].build(config)

        self.num_classes = config.dataset.num_classes

    def forward(self, x):
        image_embddings, high_res_features = self.encoder(x)
        masks = self.decoder(image_embddings, high_res_features)

        return masks
