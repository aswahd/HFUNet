from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()
        # First layer
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample layer for adjusting dimensions (if necessary)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, stride=stride, kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        # Pass input through the two conv-bn-relu layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Adjust dimensions with downsample (if needed)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual connection
        out += identity

        return out


class TrBlockAdapter(nn.Module):
    """
    Adapter block for transformer tokens.
    """

    def __init__(self, blk: nn.Module):
        super().__init__()
        dim = blk.attn.qkv.in_features
        self.adapters = nn.Sequential(
            nn.Linear(dim, 4, bias=True),
            nn.GELU(),
            nn.Linear(4, dim, bias=True),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.adapters(x)
        return x

class ConvAdapter(nn.Module):
    """
    Takes in patch embedding and transforms it to a different dimension.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Example: (64 x 64 x 64) -> (128 x 32 x 32)
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=stride,
                padding=3,
            ),
            nn.ReLU(),
            nn.Conv2d(
                64,
                out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
        )

    def forward(self, x):
        x = self.proj(x)  # B x C x H x W
        return x
        

class FpnNeck(nn.Module):
    """
    Feature Pyramid Network (FPN) neck.
    """

    def __init__(
        self,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param d_model: the dimension of the model
        :param backbone_channel_list: list of channel dimensions from the backbone
        :param kernel_size: kernel size for the convolution layers
        :param stride: stride for the convolution layers
        :param padding: padding for the convolution layers
        :param fpn_interp_model: interpolation model to use in FPN
        :param fuse_type: fusion method ('sum' or 'avg') for combining features
        :param fpn_top_down_levels: levels to have top-down features in outputs
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=dim,
                        out_channels=d_model,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            )
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"], "fuse_type must be 'sum' or 'avg'"
        self.fuse_type = fuse_type

        # Levels to have top-down features in outputs
        if fpn_top_down_levels is None:
            # Default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        out = [None] * len(self.convs)
        assert len(xs) == len(
            self.convs
        ), "Input list length must match number of conv layers"
        prev_features = None
        # Forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    recompute_scale_factor=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            out[i] = prev_features

        return out
