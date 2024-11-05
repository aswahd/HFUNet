# fmt: off
import os
import sys

sys.path.append(os.path.abspath("../SAM2UNet"))
print(sys.path)
# fmt: on


from .decoders.unet_decoder import *
from .deeplabv3_plus import *
from .encoders.hiera_vit_image_encoder import *
from .encoders.tiny_vit_image_encoder import *
from .hfunet.hfunet_hiera_base_plus import *
from .hfunet.hfunet_hiera_large import *
from .hfunet.hfunet_hiera_tiny import *
from .hfunet.hfunet_tiny_vit import *
from .manet import *
from .model_registry import HFUNET_MODEL_REGISTRY
from .PAN import *
from .sam2_unet import *
from .unet import *
from .unet_plusplus import *


def build_model(config):
    return HFUNET_MODEL_REGISTRY[config.get("model", "hfunet_tiny")](config)
