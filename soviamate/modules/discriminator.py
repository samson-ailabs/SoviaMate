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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

LEAKY_SLOPE = 0.2


class _ResBlockDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        assert (
            out_channels // in_channels == 2
        ), "Output channels must be twice the input channels"

        super().__init__()

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
        x = F.leaky_relu(x, LEAKY_SLOPE)

        x1 = self.res_conv(x)
        x2 = self.res_pool(x)

        x = x1 + x2.repeat_interleave(2, dim=1)
        x = F.leaky_relu(x, LEAKY_SLOPE)

        out = self.res_proj(x)

        return out


class _ResBlockUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:

        super().__init__()
        self.factor = in_channels // out_channels

        self.skip_upsm = nn.Upsample(scale_factor=2, mode="nearest")
        self.skip_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        self.res_upsm = nn.Upsample(scale_factor=2, mode="nearest")
        self.res_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
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
        x = F.leaky_relu(x, LEAKY_SLOPE)

        x1 = self.res_conv(x)
        x2 = self.res_upsm(x)

        x = x1 + x2[:, :: self.factor, :, :]
        x = F.leaky_relu(x, LEAKY_SLOPE)

        out = self.res_proj(x)

        return out


class SpecUnetDisc(nn.Module):
    r"""A U-Net based discriminator that operates on the spectrogram domain.

    Args:
        n_fft (int): number of Fourier bins
        win_length (int): window size
        hop_length (int): length of hop between STFT windows
    """

    def __init__(self, n_fft: int, win_length: int, hop_length: int) -> None:
        super().__init__()

        self.spectrum = T.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
        )

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=1)

        self.down1 = _ResBlockDown(16, 32)
        self.down2 = _ResBlockDown(32, 64)
        self.down3 = _ResBlockDown(64, 128)
        self.down4 = _ResBlockDown(128, 256)
        self.down5 = _ResBlockDown(256, 512)

        self.ccat1 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.ccat2 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.ccat3 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.ccat4 = nn.Conv2d(64, 32, kernel_size=1, stride=1)

        self.upsm1 = _ResBlockUp(512, 256)
        self.upsm2 = _ResBlockUp(256, 128)
        self.upsm3 = _ResBlockUp(128, 64)
        self.upsm4 = _ResBlockUp(64, 32)
        self.upsm5 = _ResBlockUp(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the discriminator.

        Args:
            x (Tensor): input tensor with shape (B, 1, T)

        Returns:
            Tensor: output tensor
        """

        x = self.spectrum(x.squeeze(1))
        x = F.pad(x, (0, -1, 0, -1))

        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2)

        x = x.contiguous()
        x0 = self.conv1(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        y0 = x5

        y1 = self.upsm1(y0)
        y1 = self.ccat1(torch.cat((x4, y1), dim=1))
        y2 = self.upsm2(y1)
        y2 = self.ccat2(torch.cat((x3, y2), dim=1))
        y3 = self.upsm3(y2)
        y3 = self.ccat3(torch.cat((x2, y3), dim=1))
        y4 = self.upsm4(y3)
        y4 = self.ccat4(torch.cat((x1, y4), dim=1))
        y5 = self.upsm5(y4)

        x = self.conv2(y5)
        x = x.flatten(start_dim=1)

        return x
