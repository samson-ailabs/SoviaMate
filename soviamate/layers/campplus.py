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

"""CAM++ speaker embedding extractor.

This module provides a clean implementation of CAM++ architecture for
speaker verification. The implementation is weight-compatible with pretrained
checkpoints from 3D-Speaker (https://github.com/modelscope/3D-Speaker).
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def _get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    """Build activation from dash-separated config (e.g. ``"batchnorm-relu"``)."""
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected activation module: {name}")
    return nonlinear


class _StatsPool(nn.Module):
    """Mean + std pooling over the time axis."""

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(B, C, T) -> (B, 2C)``. *mask*: ``(B, 1, T)``, True = valid."""
        if mask is not None:
            x = x * mask
            n = mask.sum(dim=-1)  # (B, 1)
            mean = x.sum(dim=-1) / n
            diff = (x - mean.unsqueeze(-1)) * mask
            std = (diff.pow(2).sum(dim=-1) / (n - 1)).clamp(min=1e-8).sqrt()
        else:
            mean = x.mean(dim=-1)
            std = x.std(dim=-1)
        return torch.cat([mean, std], dim=-1)


class _TDNNLayer(nn.Module):
    """Conv1d + nonlinearity."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = _get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply TDNN layer."""
        return self.nonlinear(self.linear(x))


class _CAMLayer(nn.Module):
    """Channel gate from utterance-level + segment-level context.

    Multiplies local conv output by a sigmoid gate derived from
    global mean pooling fused with segment-level pooling.
    """

    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2,
    ) -> None:
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply CAM gating."""
        y = self.linear_local(x)

        # Utterance-level context (masked mean if lengths provided)
        if mask is not None:
            utt_mean = (x * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        else:
            utt_mean = x.mean(-1, keepdim=True)

        context = utt_mean + self._seg_pooling(x)
        m = self.sigmoid(self.linear2(self.relu(self.linear1(context))))
        return y * m

    @staticmethod
    def _seg_pooling(
        x: torch.Tensor, seg_len: int = 100, stype: str = "avg"
    ) -> torch.Tensor:
        """Segment-level pooling broadcast back to original length."""
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError(f"Wrong segment pooling type: {stype}")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        return seg[..., : x.shape[-1]]


class _CAMDenseTDNNLayer(nn.Module):
    """Bottleneck → CAMLayer within a DenseNet block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        self.memory_efficient = memory_efficient
        self.nonlinear1 = _get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = _get_nonlinear(config_str, bn_channels)
        self.cam_layer = _CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            bias=bias,
        )

    def _bn_function(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear1(self.nonlinear1(x))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply bottleneck + CAMLayer with optional checkpointing."""
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self._bn_function, x, use_reentrant=False)
        else:
            x = self._bn_function(x)
        return self.cam_layer(self.nonlinear2(x), mask)


class _CAMDenseTDNNBlock(nn.ModuleList):
    """DenseNet block: each layer receives concatenated outputs of all previous."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(
                f"tdnnd{i + 1}",
                _CAMDenseTDNNLayer(
                    in_channels=in_channels + i * out_channels,
                    out_channels=out_channels,
                    bn_channels=bn_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    config_str=config_str,
                    memory_efficient=memory_efficient,
                ),
            )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply dense connectivity: concatenate input with all layer outputs."""
        for layer in self:
            x = torch.cat([x, layer(x, mask)], dim=1)
        return x


class _TransitLayer(nn.Module):
    """Channel reduction (1x1 conv) between dense blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        config_str: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        self.nonlinear = _get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transition layer."""
        return self.linear(self.nonlinear(x))


class _DenseLayer(nn.Module):
    """Pointwise (1x1) conv with configurable nonlinearity."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = _get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pointwise convolution with nonlinearity."""
        if x.dim() == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        return self.nonlinear(x)


class _BasicResBlock(nn.Module):
    """2D residual block with frequency-axis striding."""

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=(stride, 1),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D residual block with frequency-axis striding."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class _FCM(nn.Module):
    """Frequency-Channel Mapping: ``(B, F, T) → (B, C·F/8, T)``."""

    def __init__(
        self,
        feat_dim: int = 80,
        m_channels: int = 32,
        num_blocks: tuple[int, ...] = (2, 2),
    ) -> None:
        super().__init__()
        self._in_planes = m_channels

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(m_channels, num_blocks[1], stride=2)
        self.conv2 = nn.Conv2d(
            m_channels,
            m_channels,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicResBlock(self._in_planes, planes, s))
            self._in_planes = planes * _BasicResBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FCM front-end: (B, F, T) → (B, C·F/8, T)."""
        x = x.unsqueeze(1)  # (B, 1, F, T)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])


class CAMPPlus(nn.Module):
    """CAM++ speaker embedding extractor.

    Context-Aware Masking enhanced TDNN with DenseNet-style dense
    connectivity and a 2D ResNet frequency-channel mapping front-end.

    Args:
        feat_dim: Input feature dimension (mel bins).
        embedding_size: Output embedding dimension.
        growth_rate: Per-layer output channels in dense blocks.
        bn_size: Bottleneck multiplier (``bn_channels = bn_size * growth_rate``).
        init_channels: TDNN output channels after FCM.
        config_str: Nonlinear activation config (e.g. ``"batchnorm-relu"``).
        memory_efficient: Use gradient checkpointing in dense blocks.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,
    ) -> None:
        super().__init__()

        # 2D front-end: (B, feat_dim, T) → (B, C, T)
        self.head = _FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        # 1D backbone stored as Sequential for checkpoint compatibility
        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        _TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels

        # 3 dense blocks with transit layers
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            self.xvector.add_module(
                f"block{i + 1}",
                _CAMDenseTDNNBlock(
                    num_layers=num_layers,
                    in_channels=channels,
                    out_channels=growth_rate,
                    bn_channels=bn_size * growth_rate,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    config_str=config_str,
                    memory_efficient=memory_efficient,
                ),
            )
            channels += num_layers * growth_rate

            self.xvector.add_module(
                f"transit{i + 1}",
                _TransitLayer(
                    channels,
                    channels // 2,
                    bias=False,
                    config_str=config_str,
                ),
            )
            channels //= 2

        # Pooling + embedding projection
        self.xvector.add_module("out_nonlinear", _get_nonlinear(config_str, channels))
        self.xvector.add_module("stats", _StatsPool())
        self.xvector.add_module(
            "dense",
            _DenseLayer(channels * 2, embedding_size, config_str="batchnorm_"),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract speaker embedding.

        Args:
            x: Fbank features ``(B, feat_dim, T)``.
            lengths: Valid frame lengths ``(B,)``. Masks padded frames
                in temporal pooling (StatsPool, CAMLayer).

        Returns:
            Utterance embedding ``(B, embedding_size)``.
        """
        x = self.head(x)

        # Stride-2 TDNN, then build mask at the downsampled resolution
        mask = None
        x = self.xvector.tdnn(x)
        if lengths is not None:
            lengths = (lengths + 1) // 2  # ceil div for stride=2
            indices = torch.arange(x.size(-1), device=x.device)
            mask = (indices < lengths.unsqueeze(1)).unsqueeze(1)  # (B, 1, T)

        for name, module in self.xvector.named_children():
            if name == "tdnn":
                continue
            elif isinstance(module, (_CAMDenseTDNNBlock, _StatsPool)):
                x = module(x, mask)
            else:
                x = module(x)

        return x
