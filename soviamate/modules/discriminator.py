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

"""Discriminators for adversarial training"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.2


class _ResBlockDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        assert (
            out_channels // in_channels == 2
        ), "Output channels must be twice the input channels"

        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=LRELU_SLOPE)

        self.skip_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.skip_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        self.res_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.res_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.res_proj = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the downsample block.

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """

        x1 = self._skip_connection(x)
        x2 = self._residual_connection(x)

        x = F.normalize(x1 + 0.4 * x2, p=2, dim=1)

        return x

    def _skip_connection(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.skip_pool(x)
        x2 = self.skip_conv(x1)

        out = torch.cat((x1, x2), dim=1)

        return out

    def _residual_connection(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(x)

        x1 = self.res_conv(x)
        x2 = self.res_pool(x)

        x = x1 + x2.repeat_interleave(2, dim=1)
        x = self.leaky_relu(x)

        out = self.res_proj(x)

        return out


class _ResBlockUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:

        super().__init__()
        self.factor = in_channels // out_channels
        self.leaky_relu = nn.LeakyReLU(negative_slope=LRELU_SLOPE)

        self.skip_upsm = nn.Upsample(scale_factor=2, mode="nearest")
        self.skip_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        self.res_upsm = nn.Upsample(scale_factor=2, mode="nearest")
        self.res_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.res_proj = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the upsample block.

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """

        x1 = self._skip_connection(x)
        x2 = self._residual_connection(x)

        x = F.normalize(x1 + 0.4 * x2, p=2, dim=1)

        return x

    def _skip_connection(self, x: torch.Tensor) -> torch.Tensor:
        x = self.skip_conv(x)
        out = self.skip_upsm(x)
        return out

    def _residual_connection(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(x)

        x1 = self.res_conv(x)
        x2 = self.res_upsm(x)

        x = x1 + x2[:, :: self.factor, :, :]
        x = self.leaky_relu(x)

        out = self.res_proj(x)

        return out


class SpecUnetDisc(nn.Module):
    r"""A U-Net based discriminator that operates on the spectrogram domain.

    Args:
        n_ffts (List[int]): list of FFT sizes for each spectrogram
        win_lengths (List[int]): list of window lengths for each spectrogram
        hop_lengths (List[int]): list of hop lengths for each spectrogram
    """

    def __init__(
        self, n_ffts: List[int], win_lengths: List[int], hop_lengths: List[int]
    ) -> None:
        super().__init__()

        assert (
            len(n_ffts) == len(win_lengths) == len(hop_lengths)
        ), "Lengths of n_ffts, win_lengths, and hop_lengths must be the same"

        self.spectrograms = nn.ModuleList()
        for n_fft, win_length, hop_length in zip(n_ffts, win_lengths, hop_lengths):
            self.spectrograms.append(
                T.Spectrogram(
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    power=None,
                )
            )

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.down1 = _ResBlockDown(64, 128)
        self.down2 = _ResBlockDown(128, 256)
        self.down3 = _ResBlockDown(256, 512)
        self.down4 = _ResBlockDown(512, 1024)

        self.ccat1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.ccat2 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.ccat3 = nn.Conv2d(256, 128, kernel_size=1, stride=1)

        self.upsm1 = _ResBlockUp(1024, 512)
        self.upsm2 = _ResBlockUp(512, 256)
        self.upsm3 = _ResBlockUp(256, 128)
        self.upsm4 = _ResBlockUp(128, 64)

    def _forward_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2)

        x = x.contiguous()
        x0 = self.conv1(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        y0 = x4

        y1 = self.upsm1(y0)
        y1 = self.ccat1(torch.cat((x3, y1), dim=1))
        y2 = self.upsm2(y1)
        y2 = self.ccat2(torch.cat((x2, y2), dim=1))
        y3 = self.upsm3(y2)
        y3 = self.ccat3(torch.cat((x1, y3), dim=1))
        y4 = self.upsm4(y3)

        x = self.conv2(y4)
        x = x.flatten(start_dim=1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the discriminator.

        Args:
            x (Tensor): waveform input tensor, shape (B, 1, T)

        Returns:
            Tensor: latent output tensor, shape (B, T')
        """

        outs = []
        for spectrogram in self.spectrograms:
            spec = spectrogram(x.squeeze(1))[:, :-1, :-1]
            outs.append(self._forward_spectrogram(spec))

        outs = torch.cat(outs, dim=1)

        return outs
