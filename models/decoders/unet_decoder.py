from typing import List, Optional

import torch
from torch import nn

from utils import DotDict

from .registry import register_mask_decoder


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(self.relu(self.upconv(x)))


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.classifier(x)


class UNetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_in_channels: int,
        stride: int = 1,
    ):
        super(UNetDecoderBlock, self).__init__()

        if stride == 1:
            self.upconv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            # Transposed convolution for upsampling (increasing spatial resolution)
            self.upconv = UpConvBlock(
                in_channels, out_channels, kernel_size=2, stride=2, padding=0
            )

        # Two convolutional layers after upsampling and concatenation
        self.conv1 = nn.Conv2d(
            out_channels + out_channels, out_channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Change the output dimension of the skip connection to match the output of the decoder
        self.conv_skip = nn.Conv2d(skip_in_channels, out_channels, kernel_size=1)

    def forward(self, x, skip_features):
        # Upsample the input feature map
        x = self.upconv(x)
        skip_features = self.conv_skip(skip_features)

        # Concatenate the skip connection features from the encoder
        x = torch.cat(
            [x, skip_features], dim=1
        )  # Concatenate along the channel dimension

        # Pass through the two convolutional layers with ReLU and BatchNorm
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        encoder_embed_dims: List[int] = [64, 64, 128, 160, 320, 256],
        strides: List[int] = [2, 2, 1, 1, 1],
        num_classes: int = 1,
        decoder_output_dim: int = 256,
        deep_supervision: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()
        self.num_classes = num_classes
        self.decoder_output_dim = decoder_output_dim

        self.encoder_embed_dims = encoder_embed_dims
        # Decoder layers -> Change the output of the transformer to encoder channel dim at each layer
        self.decoder_layers = nn.ModuleList()

        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.deep_supervision_layers = nn.ModuleList()

        # Encoder dims: [64, 64, 128, 160, 320, 256]
        # Decoder: 256 -> 320 -> 160 -> 128 -> 64 -> 64
        encoder_out_dims_reversed: List[int] = list(reversed(self.encoder_embed_dims))
        skip_conn_input_dims = list(reversed(self.encoder_embed_dims[:-1]))
        decoder_output_dims = [self.decoder_output_dim] * (
            len(self.encoder_embed_dims) - 1
        )  # Remove neck
        decoder_input_dims = [encoder_out_dims_reversed[0]] + decoder_output_dims[:-1]
        strides.reverse()

        for i, (in_channels, out_channels, skip_in_channels, stride) in enumerate(
            zip(decoder_input_dims, decoder_output_dims, skip_conn_input_dims, strides)
        ):
            self.decoder_layers.append(
                UNetDecoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    skip_in_channels=skip_in_channels,
                    stride=stride,
                )
            )

            if self.deep_supervision and i < len(encoder_out_dims_reversed) - 1:
                self.deep_supervision_layers.append(
                    SegmentationHead(out_channels, num_classes)
                )

        # Upscale the output to the original image resolution
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                self.decoder_output_dim,
                self.decoder_output_dim,
                kernel_size=2,
                stride=2,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.decoder_output_dim,
                self.decoder_output_dim,
                kernel_size=2,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(self.decoder_output_dim, self.decoder_output_dim, kernel_size=1),
        )

        self.segmentation_head = SegmentationHead(self.decoder_output_dim, num_classes)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
            mask.
        """
        assert high_res_features is not None, "Pass encoder features."
        # high_res_feats: [96, 192, 384, 768] -> [768, 384, 192, 96]
        high_res_features.reverse()

        src = image_embeddings
        # Generate masks (possibly) at each decoder layer
        masks_out: List[torch.Tensor] = []
        for i, decoder_layer in enumerate(self.decoder_layers):
            src = decoder_layer(src, high_res_features[i])  # (b, c, h, w)
            if self.deep_supervision:
                masks_out.append(self.deep_supervision_layers[i](src))

        # Output mask at original image resolution
        src = self.output_upscaling(src)
        masks_out.append(self.segmentation_head(src))

        return masks_out


@register_mask_decoder("mask_decoder")
class MaskDecoderFactory(object):
    @staticmethod
    def build(
        config: DotDict,
    ) -> MaskDecoder:
        """Build mask decoder with skip connections."""

        encoder_embed_dims: List[int] = config.image_encoder.embed_dims
        strides: List[int] = config.image_encoder.strides
        num_classes: int = config.decoder.num_classes
        decoder_output_dim: int = config.decoder.get("decoder_output_dim", 64)
        return MaskDecoder(
            encoder_embed_dims=encoder_embed_dims,
            strides=strides,
            num_classes=num_classes,
            decoder_output_dim=decoder_output_dim,
            deep_supervision=config.get("deep_supervision", False),
        )
