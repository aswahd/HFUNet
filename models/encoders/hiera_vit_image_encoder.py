from functools import partial
from typing import List

import torch
import torch.nn as nn

from mppm.models.sam2.modeling.backbones.hieradet import Hiera
from mppm.models.sam2.modeling.backbones.image_encoder import (
    FpnNeck,
)
from mppm.models.sam2.modeling.position_encoding import PositionEmbeddingSine
from utils import DotDict

from ..common import ConvAdapter
from .registry import register_image_encoder


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # Forward through backbone
        features, _ = self.neck(self.trunk(x))
        image_embeddings, features = features[-1], features[:-1]
        return image_embeddings, features

    def freeze_pretrained_parameters(self):
        ignore_keys = ["adapter", "neck"]
        for name, param in self.named_parameters():
            if any([key in name for key in ignore_keys]):
                continue
            param.requires_grad = False

    def load_pretrained_weights(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
        # Extract image encoder state dict
        state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in state_dict.items()
            if "image_encoder." in k
        }

        # Ignore keys for adapter layers
        ignore_keys = ["adapters", "patch_adapter"]
        except_keys = []
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        missing_keys = {
            k for k in missing_keys if not any([key in k for key in ignore_keys])
        }

        if missing_keys:
            print("Missing keys: ", missing_keys)
            raise RuntimeError()

        unexpected_keys = {
            k for k in unexpected_keys if not any([key in k for key in except_keys])
        }

        if unexpected_keys:
            print("Unexpected keys: ", unexpected_keys)
            raise RuntimeError()

        print(
            "{} loaded checkpoint from {} successfully.".format(
                self.__class__.__name__,
                checkpoint_path,
            )
        )


class HieraTinyImageEncoder(ImageEncoder):
    def __init__(self):
        super().__init__()

        self.scalp = 1

        self.trunk = Hiera(
            embed_dim=96,
            num_heads=1,
            stages=[1, 2, 7, 2],
            global_att_blocks=[5, 7, 9],
            window_pos_embed_bkg_spatial_size=[7, 7],
        )

        self.neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10_000,
            ),
            d_model=256,
            backbone_channel_list=[768, 384, 192, 96],
            fpn_top_down_levels=[0, 1, 2, 3],
            fpn_interp_model="nearest",
        )

        self.freeze_pretrained_parameters()


class HieraTinyImageEncoderFactory(HieraTinyImageEncoder):
    @staticmethod
    def build(config: DotDict) -> HieraTinyImageEncoder:
        return HieraTinyImageEncoder()


class ViTEncoderWithAdapters(nn.Module):
    """
    Vision Transformer Encoder with Adapter Layers.

    This class integrates adapter layers into a Vision Transformer (ViT) backbone.
    The adapter layers are added to specific blocks of the ViT to enhance its performance.

    Args:
        backbone (nn.Module): The Vision Transformer backbone model.
        patch_embed_dim (int): Dimension of the patch embeddings. Default is 96.
        stages (List[int]): Number of blocks in each stage of the ViT. Default is [1, 2, 7, 2].
        backbone_channel_list (List[int]): List of channel dimensions for each stage. Default is [768, 384, 192, 96].
        strides (List[int]): List of stride values for each stage. Default is [1, 2, 4, 8].
        adapter_hidden_dim (int): Hidden dimension for the adapter layers. Default is 16.
        blocks_with_adapters (List[int]): List of block indices where adapters are added. Default is [10, 11].
    """

    def __init__(
        self,
        backbone: nn.Module,
        patch_embed_dim: int = 96,
        stages: List[int] = [1, 2, 7, 2],
        backbone_channel_list: List[int] = [768, 384, 192, 96],
        strides: List[int] = [1, 2, 4, 8],
        adapter_hidden_dim: int = 16,
        blocks_with_adapters: List[int] = [-4, -3, -2, -1],
    ):
        super().__init__()
        self.backbone = backbone
        self.adapters = nn.ModuleList()
        backbone_channel_list.reverse()
        self.backbone.trunk.patch_embed.register_forward_hook(
            self._set_patch_embeddings
        )

        vit_block_dims, block_strides = [], []
        for stage, dim, stride in zip(stages, backbone_channel_list, strides):
            vit_block_dims.extend([dim] * stage)
            block_strides.extend([stride] * stage)

        self.patch_adapter = nn.Sequential(
            nn.Linear(patch_embed_dim, adapter_hidden_dim), nn.ReLU()
        )

        # Negative block indices are counted from the end
        blocks_with_adapters = [
            i if i >= 0 else len(vit_block_dims) + i for i in blocks_with_adapters
        ]

        for i, (block, dim, stride) in enumerate(
            zip(self.backbone.trunk.blocks, vit_block_dims, block_strides)
        ):
            if i not in blocks_with_adapters:
                continue

            # Add adapter layer to ViT block
            adapter = ConvAdapter(
                in_channels=adapter_hidden_dim,
                out_channels=dim,
                stride=stride,
            )
            adapter.get_patch_embedding = self._get_patch_embeddings
            self.adapters.append(adapter)
            block.register_forward_hook(
                partial(self._transform_output_with_adapter, adapter)
            )

    def _set_patch_embeddings(self, module, input, output):
        """
        Hook to set patch embeddings.

        Args:
            module (nn.Module): The module to which the hook is registered.
            input (torch.Tensor): The input tensor to the module.
            output (torch.Tensor): The output tensor from the module.
        """
        self._patch_embeddings = self.patch_adapter(output)

    def _get_patch_embeddings(self):
        """
        Get the patch embeddings.

        Returns:
            torch.Tensor: The patch embeddings.
        """
        return self._patch_embeddings

    def _transform_output_with_adapter(self, adapter, module, input, output):
        """
        Hook to adapt the output of a ViT block using the adapter.

        Args:
            adapter (ConvAdapter): The adapter layer.
            module (nn.Module): The module to which the hook is registered.
            input (torch.Tensor): The input tensor to the module.
            output (torch.Tensor): The output tensor from the module.

        Returns:
            torch.Tensor: The adapted output tensor.
        """
        patch_embed = adapter.get_patch_embedding().permute(0, 3, 1, 2)
        prompt = adapter(patch_embed).permute(0, 2, 3, 1)
        return prompt + output

    def load_pretrained_weights(self, checkpoint_path: str):
        self.backbone.load_pretrained_weights(checkpoint_path)

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)


HieraTinyBlockwiseAdapter = ViTEncoderWithAdapters


@register_image_encoder("hiera_tiny_blockwise_adapter")
class TinyViTBlockwiseAdapterFactory:
    @staticmethod
    def build(config):
        backbone = HieraTinyImageEncoderFactory.build(config)
        return HieraTinyBlockwiseAdapter(backbone)


class HieraBasePlusImageEncoder(ImageEncoder):
    def __init__(self):
        super().__init__()

        # configs
        self.scalp = 1
        self.trunk = Hiera(
            embed_dim=112,
            num_heads=2,
            stages=[2, 3, 16, 3],
            global_att_blocks=[12, 16, 20],
            window_pos_embed_bkg_spatial_size=[14, 14],
        )

        self.neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10_000,
            ),
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112],
            fpn_top_down_levels=[0, 1, 2, 3],
            fpn_interp_model="bilinear",
        )

        self.freeze_pretrained_parameters()


class HieraTBasePlusBlockwiseAdapter(ViTEncoderWithAdapters):
    def __init__(self, backbone):
        super().__init__(
            backbone=backbone,
            patch_embed_dim=112,
            stages=[2, 3, 16, 3],
            backbone_channel_list=[896, 448, 224, 112],
            strides=[1, 2, 4, 8],
            adapter_hidden_dim=16,
            blocks_with_adapters=[-3, -2, -1],
        )


class HieraBasePlusImageEncoderFactory:
    @staticmethod
    def build(config: DotDict) -> HieraBasePlusImageEncoder:
        return HieraBasePlusImageEncoder()


@register_image_encoder("hiera_base_plus_blockwise_adapter")
class HieraTBasePlusBlockwiseAdapterFactory:
    @staticmethod
    def build(config: DotDict) -> HieraTBasePlusBlockwiseAdapter:
        backbone = HieraBasePlusImageEncoderFactory.build(config)
        return HieraTBasePlusBlockwiseAdapter(backbone)


class HieraLargeImageEncoder(ImageEncoder):
    def __init__(self):
        super().__init__()

        # configs
        self.scalp = 1
        self.trunk = Hiera(
            embed_dim=144,
            num_heads=2,
            stages=[2, 6, 36, 4],
            global_att_blocks=[23, 33, 43],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 16, 8],
        )

        self.neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10_000,
            ),
            d_model=256,
            backbone_channel_list=[1152, 576, 288, 144],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )

        self.freeze_pretrained_parameters()


@register_image_encoder("hfunet_large_encoder")
class HieraLargeImageEncoderFactory:
    @staticmethod
    def build(config: DotDict) -> HieraLargeImageEncoder:
        return HieraLargeImageEncoder()


class HieraLargeBlockwiseAdapter(ViTEncoderWithAdapters):
    def __init__(self, backbone):
        super().__init__(
            backbone=backbone,
            patch_embed_dim=144,
            stages=[2, 6, 36, 4],
            backbone_channel_list=[1152, 576, 288, 144],
            strides=[1, 2, 4, 8],
            adapter_hidden_dim=16,
            blocks_with_adapters=[40, 41, 42],
        )


@register_image_encoder("hiera_large_blockwise_adapter")
class HieraLargeAdapterFactory:
    @staticmethod
    def build(config) -> HieraLargeBlockwiseAdapter:
        backbone = HieraLargeImageEncoderFactory.build(config)
        return HieraLargeBlockwiseAdapter(backbone)


if __name__ == "__main__":
    encoder = HieraTinyImageEncoder()
    x = torch.rand(1, 3, 1024, 1024)

    image_embed, high_res_feats = encoder(x)

    for f in high_res_feats:
        print(f.shape)
