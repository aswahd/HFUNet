# Code adapted from https://github.com/ChaoningZhang/MobileSAM/blob/master/mobile_sam/modeling/image_encoder.py
import itertools
import math
from functools import partial
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath as TimmDropPath
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model

from ..common import (
    ConvAdapter,
    FpnNeck,
)
from .registry import register_image_encoder


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1
    ):
        super().__init__()
        self.add_module(
            "c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f"(drop_prob={self.drop_prob})"
        return msg


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(
            self.hidden_chans,
            self.hidden_chans,
            ks=3,
            stride=1,
            pad=1,
            groups=self.hidden_chans,
        )
        self.act2 = activation()

        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 2
        if out_dim == 320 or out_dim == 448 or out_dim == 576:
            stride_c = 1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=(14, 14),
    ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False
        )

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.register_buffer(
                "ab",
                self.attention_biases[:, self.attention_bias_idxs],
                persistent=False,
            )

    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3
        )
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs]
            if self.training
            else self.ab
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class TinyViTBlock(nn.Module):
    r"""TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        activation=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(
            dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=mlp_activation,
            drop=drop,
        )

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = (
                x.view(B, nH, self.window_size, nW, self.window_size, C)
                .transpose(2, 3)
                .reshape(B * nH * nW, self.window_size * self.window_size, C)
            )
            x = self.attn(x)
            # window reverse
            x = (
                x.view(B, nH, nW, self.window_size, self.window_size, C)
                .transpose(2, 3)
                .reshape(B, pH, pW, C)
            )

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    """A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        local_conv_size=3,
        activation=nn.GELU,
        out_dim=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    local_conv_size=local_conv_size,
                    activation=activation,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TinyViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            resolution=img_size,
            activation=activation,
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=(
                    patches_resolution[0]
                    // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                    patches_resolution[1]
                    // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                ),
                #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                #                     patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation,
            )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )

            self.layers.append(layer)

        self.high_res_feats = [None] * (self.num_layers + 1)  # Include patch embedding

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

        self.freeze_pretrained_parameters()

    def freeze_pretrained_parameters(self):
        for name, param in self.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
        # print("LR SCALES:", lr_scales)

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))

        assert i == depth

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained_weights(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Extract image encoder state dict
        state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in state_dict.items()
            if "image_encoder." in k
        }

        # Ignore keys for adapter layers
        ignore_keys = ["adapters", "patch_adapter"]
        except_keys = ["norm_head", "head"]
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        missing_keys = {
            k for k in missing_keys if not any([key in k for key in ignore_keys])
        }

        if missing_keys:
            print(missing_keys)
            raise RuntimeError()

        unexpected_keys = {
            k for k in unexpected_keys if not any([key in k for key in except_keys])
        }

        if unexpected_keys:
            print(unexpected_keys)
            raise RuntimeError()

        print(
            "{} loaded checkpoint from {} successfully.".format(
                self.__class__.__name__,
                checkpoint_path,
            )
        )

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"attention_biases"}

    def forward_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)
        self.high_res_feats[0] = x  # Patch embedding

        start_i = 0
        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
            b, n, c = x.shape
            h = w = int(math.sqrt(n))
            self.high_res_feats[i + 1] = x.permute(0, 2, 1).view(b, c, h, w)

        B, T, C = x.size()
        H = W = int(math.sqrt(T))
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.high_res_feats + [x]


class TinyVitWithFpnNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        backbone_channel_list: List[int] = [320, 160, 128],
        stages: List[int] = [2, 6, 2],
        strides: List[int] = [2, 4, 4],
        patch_embed_dim: int = 64,
        adapter_hidden_dim=16,
        blocks_with_adapters: List[int] = list(range(3)),
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.adapters = nn.ModuleList()
        backbone_channel_list.reverse()

        self.trunk.patch_embed.register_forward_hook(self._set_patch_embeddings)
        self.patch_adapter = nn.Sequential(nn.Linear(patch_embed_dim, adapter_hidden_dim), nn.ReLU())

        # Extract all ViT Blocks
        vit_blocks = []
        for layer in self.trunk.layers[1:]:  # the first layer is a conv layer
            vit_blocks.extend(layer.blocks)

        vit_block_dims, block_strides = [], []
        for stage, dim, stride in zip(stages, backbone_channel_list, strides):
            vit_block_dims.extend([dim] * stage)
            block_strides.extend([stride] * stage)

        # Negative block indices are counted from the end
        blocks_with_adapters = [
            i if i >= 0 else len(vit_block_dims) + i for i in blocks_with_adapters
        ]

        for i, (block, dim, stride) in enumerate(
            zip(vit_blocks, vit_block_dims, block_strides)
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

    def _set_patch_embeddings(self, module, input, patch_embeds):
        """
        Hook to set patch embeddings.

        Args:
            module (nn.Module): The module to which the hook is registered.
            input (torch.Tensor): The input tensor to the module.
            patch_embeds (torch.Tensor): The output tensor of patch_embed.
        """
        self._patch_embeddings = self.patch_adapter(patch_embeds)

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
        # Get patch embeddings
        patch_embeds = adapter.get_patch_embedding().permute(
            0, 3, 1, 2
        )  # B x C x H x W
        prompt = (
            adapter(patch_embeds).flatten(2).permute(0, 2, 1)
        )  # B x C x H x W -> B x T x C
        modfied_output = output + prompt

        return modfied_output

    def load_pretrained_weights(self, checkpoint_path: str):
        self.trunk.load_pretrained_weights(checkpoint_path)

    def forward(self, *args, **kwargs):
        backbone_features = self.trunk(*args, **kwargs)
        features = self.neck(backbone_features)
        image_embeddings, features = features[-1], features[:-1]
        return image_embeddings, features


class TinyViTFactory:
    @staticmethod
    def build(config):
        return TinyViT(
            img_size=config.dataset.image_size,
            in_chans=3,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )


@register_image_encoder("tiny_vit_with_fpn_neck")
class TinyVitWithFpnNeckFactory(TinyViTFactory):
    @staticmethod
    def build(config):
        trunk = TinyViTFactory.build(config)
        neck = FpnNeck(
            d_model=256,
            backbone_channel_list=[
                256,
                320,
                320,
                160,
                128,
                64,
            ],  # Patch embedding to neck
            fpn_top_down_levels=[0, 1],
            fpn_interp_model="bilinear",
        )

        return TinyVitWithFpnNeck(trunk, neck)


_checkpoint_url_format = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/{}.pth"
)
_provided_checkpoints = {
    "tiny_vit_5m_224": "tiny_vit_5m_22kto1k_distill",
    "tiny_vit_11m_224": "tiny_vit_11m_22kto1k_distill",
    "tiny_vit_21m_224": "tiny_vit_21m_22kto1k_distill",
    "tiny_vit_21m_384": "tiny_vit_21m_22kto1k_384_distill",
    "tiny_vit_21m_512": "tiny_vit_21m_22kto1k_512_distill",
}


def register_tiny_vit_model(fn):
    """Register a TinyViT model
    It is a wrapper of `register_model` with loading the pretrained checkpoint.
    """

    def fn_wrapper(pretrained=False, **kwargs):
        model = fn()
        if pretrained:
            model_name = fn.__name__
            assert (
                model_name in _provided_checkpoints
            ), f"Sorry that the checkpoint `{model_name}` is not provided yet."
            url = _checkpoint_url_format.format(_provided_checkpoints[model_name])
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url,
                map_location="cpu",
                check_hash=False,
            )
            model.load_state_dict(checkpoint["model"])

        return model

    # rename the name of fn_wrapper
    fn_wrapper.__name__ = fn.__name__
    return register_model(fn_wrapper)


# @register_tiny_vit_model
def tiny_vit_11m_224(img_size=1024, drop_path_rate=0.1):
    return TinyViT(
        img_size=img_size,
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


# @register_tiny_vit_model
def tiny_vit_21m_224(img_size=1024, drop_path_rate=0.2):
    return TinyViT(
        img_size=img_size,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


# @register_tiny_vit_model
def tiny_vit_21m_384(img_size=1024, drop_path_rate=0.1):
    return TinyViT(
        img_size=img_size,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[12, 12, 24, 12],
        drop_path_rate=drop_path_rate,
    )


# @register_tiny_vit_model
def tiny_vit_21m_512(img_size=1024, drop_path_rate=0.1):
    return TinyViT(
        img_size=img_size,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=drop_path_rate,
    )