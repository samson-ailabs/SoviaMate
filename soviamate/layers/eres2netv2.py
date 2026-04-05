# Copyright (c) 2025, Son Dang Dinh. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ERes2NetV2 speaker embedding extractor.

This module provides a clean implementation of ERes2NetV2 architecture for
speaker verification. The implementation is weight-compatible with pretrained
checkpoints from 3D-Speaker (https://github.com/modelscope/3D-Speaker).
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClippedReLU(nn.Hardtanh):
    """ReLU activation with upper bound clipping at 20."""

    def __init__(self, inplace: bool = False) -> None:
        """Initialize ClippedReLU.

        Args:
            inplace (bool): Whether to perform the operation in-place.
        """
        super().__init__(min_val=0, max_val=20, inplace=inplace)


class TAP(nn.Module):
    """Temporal Average Pooling.

    Computes the mean across the temporal dimension.
    """

    def __init__(self, **_kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, T).

        Returns:
            Tensor: Pooled tensor of shape (B, C * H).
        """
        return x.mean(dim=-1).flatten(start_dim=1)


class TSTP(nn.Module):
    """Temporal Statistics Pooling with optional masking."""

    def __init__(self, **_kwargs) -> None:
        super().__init__()

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, T).
            lengths (Tensor, optional): Valid lengths of shape (B,).

        Returns:
            Tensor: Pooled tensor of shape (B, 2 * C * H).
        """
        if lengths is None:
            mean = x.mean(dim=-1)
            std = (x.var(dim=-1) + 1e-8).sqrt()
        else:
            # Create mask: (B, 1, 1, T)
            max_len = x.size(-1)
            mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1)
            mask = (mask < lengths.unsqueeze(1)).float().unsqueeze(1).unsqueeze(1)

            # Masked mean and std
            x_masked = x * mask
            count = mask.sum(dim=-1).clamp(min=1)
            mean = x_masked.sum(dim=-1) / count
            var = ((x - mean.unsqueeze(-1)) ** 2 * mask).sum(dim=-1) / count
            std = (var + 1e-8).sqrt()

        return torch.cat([mean.flatten(1), std.flatten(1)], dim=1)


class TSDP(nn.Module):
    """Temporal Standard Deviation Pooling."""

    def __init__(self, **_kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, T).

        Returns:
            Tensor: Pooled tensor of shape (B, C * H).
        """
        return (x.var(dim=-1) + 1e-8).sqrt().flatten(start_dim=1)


PoolingType = Literal["TAP", "TSTP", "TSDP"]
POOLING_LAYERS = {"TAP": TAP, "TSTP": TSTP, "TSDP": TSDP}


class AFF(nn.Module):
    """Attentional Feature Fusion.

    Args:
        channels (int): Number of input channels for each feature map.
        reduction (int): Channel reduction ratio for the attention bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = channels // reduction
        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuse two feature maps with learned attention.

        Args:
            x (Tensor): First feature map of shape (B, C, H, W).
            y (Tensor): Second feature map of shape (B, C, H, W).

        Returns:
            Tensor: Fused feature map of shape (B, C, H, W).
        """
        att = 1 + torch.tanh(self.local_att(torch.cat([x, y], dim=1)))
        return x * att + y * (2 - att)


class Res2Block(nn.Module):
    """Res2Net bottleneck block with multi-scale feature extraction.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Base channel count (output channels = channels * expansion).
        stride (int): Spatial stride for downsampling.
        base_width (int): Width multiplier for computing internal width.
        scale (int): Number of parallel convolution branches.
        expansion (int): Output channel expansion factor.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        base_width: int = 26,
        scale: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()

        width = channels * base_width // 64
        out_channels = channels * expansion

        self.conv1 = nn.Conv2d(in_channels, width * scale, 1, stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.convs = nn.ModuleList(
            [nn.Conv2d(width, width, 3, padding=1, bias=False) for _ in range(scale)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scale)])

        self.conv3 = nn.Conv2d(width * scale, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = ClippedReLU(inplace=True)
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C', H', W').
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        splits = torch.split(out, self.width, dim=1)

        sp = self.relu(self.bns[0](self.convs[0](splits[0])))

        outs = [sp]
        for i in range(1, len(splits)):
            sp = self.relu(self.bns[i](self.convs[i](sp + splits[i])))
            outs.append(sp)

        out = self.bn3(self.conv3(torch.cat(outs, dim=1)))
        return self.relu(out + self.shortcut(identity))


class Res2BlockAFF(nn.Module):
    """Res2Net block with Attentional Feature Fusion.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Base channel count (output channels = channels * expansion).
        stride (int): Spatial stride for downsampling.
        base_width (int): Width multiplier for computing internal width.
        scale (int): Number of parallel convolution branches.
        expansion (int): Output channel expansion factor.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        base_width: int = 26,
        scale: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()

        width = channels * base_width // 64
        out_channels = channels * expansion

        self.conv1 = nn.Conv2d(in_channels, width * scale, 1, stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.convs = nn.ModuleList(
            [nn.Conv2d(width, width, 3, padding=1, bias=False) for _ in range(scale)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scale)])
        self.fuse_models = nn.ModuleList(
            [AFF(width, reduction=4) for _ in range(scale - 1)]
        )

        self.conv3 = nn.Conv2d(width * scale, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = ClippedReLU(inplace=True)
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C', H', W').
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        splits = torch.split(out, self.width, dim=1)

        sp = self.relu(self.bns[0](self.convs[0](splits[0])))

        outs = [sp]
        for i in range(1, len(splits)):
            sp = self.fuse_models[i - 1](sp, splits[i])
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            outs.append(sp)

        out = self.bn3(self.conv3(torch.cat(outs, dim=1)))
        return self.relu(out + self.shortcut(identity))


class ERes2NetV2(nn.Module):
    """ERes2NetV2 speaker embedding extractor.

    Args:
        feat_dim (int): Input mel spectrogram dimension.
        embedding_size (int): Output embedding dimension.
        channels (int): Base channel width.
        num_blocks (tuple[int, ...]): Number of blocks per stage (stage1, stage2, stage3, stage4).
        base_width (int): Width factor for Res2Net blocks.
        scale (int): Number of scales in Res2Net blocks.
        expansion (int): Channel expansion factor.
        pooling (PoolingType): Temporal pooling type ("TAP", "TSTP", or "TSDP").
        two_emb_layer (bool): Whether to use two-layer embedding head with batch norm.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        channels: int = 64,
        num_blocks: tuple[int, ...] = (3, 4, 6, 3),
        base_width: int = 26,
        scale: int = 2,
        expansion: int = 2,
        pooling: PoolingType = "TSTP",
        two_emb_layer: bool = False,
    ) -> None:
        super().__init__()

        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.two_emb_layer = two_emb_layer

        self._in_planes = channels
        self._base_width = base_width
        self._scale = scale
        self._expansion = expansion

        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.layer1 = self._make_layer(Res2Block, channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(Res2Block, channels * 2, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(
            Res2BlockAFF, channels * 4, num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            Res2BlockAFF, channels * 8, num_blocks[3], stride=2
        )

        self.layer3_ds = nn.Conv2d(
            channels * 4 * expansion,
            channels * 8 * expansion,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.fuse34 = AFF(channels * 8 * expansion, reduction=4)

        stats_dim = (feat_dim // 8) * channels * 8 * expansion
        n_stats = 1 if pooling in ("TAP", "TSDP") else 2

        self.pool = POOLING_LAYERS[pooling]()
        self.seg_1 = nn.Linear(stats_dim * n_stats, embedding_size)

        if two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(
        self,
        block: Res2Block | Res2BlockAFF,
        channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a stage with the specified number of blocks.

        Args:
            block (Res2Block | Res2BlockAFF): Block class to use.
            channels (int): Base channel count for this stage.
            num_blocks (int): Number of blocks in this stage.
            stride (int): Stride for the first block (others use stride=1).

        Returns:
            nn.Sequential: Sequential container of blocks.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(
                block(
                    in_channels=self._in_planes,
                    channels=channels,
                    stride=stride if i == 0 else 1,
                    base_width=self._base_width,
                    scale=self._scale,
                    expansion=self._expansion,
                )
            )
            self._in_planes = channels * self._expansion
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract speaker embedding from mel spectrogram.

        Args:
            x (Tensor): Mel spectrogram of shape (B, D, T).
            lengths (Tensor, optional): Valid frame lengths of shape (B,).

        Returns:
            Tensor: Utterance embedding of shape (B, embedding_size).
        """
        x = x.unsqueeze(1)  # (B, 1, D, T)
        x = F.relu(self.bn1(self.conv1(x)))

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.fuse34(x4, self.layer3_ds(x3))

        # Downsample lengths to match pooling dimension (8x total stride)
        pool_lengths = None
        if lengths is not None:
            pool_lengths = (lengths / 8).ceil().long().clamp(max=x.size(-1))

        x = self.pool(x, pool_lengths)
        x = self.seg_1(x)

        if self.two_emb_layer:
            x = F.relu(x)
            x = self.seg_bn_1(x)
            x = self.seg_2(x)

        return x
