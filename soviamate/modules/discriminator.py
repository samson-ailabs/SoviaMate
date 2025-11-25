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

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.2
SEGMENT_SIZE = 16384


class _ResBlockDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        assert out_channels // in_channels == 2, (
            "Output channels must be twice the input channels"
        )

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

        assert len(n_ffts) == len(win_lengths) == len(hop_lengths), (
            "Lengths of n_ffts, win_lengths, and hop_lengths must be the same"
        )

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


class SpectralDiscriminator(nn.Module):
    r"""A spectral discriminator that operates on the spectrogram domain.

    Args:
        n_fft (int): FFT size for STFT
        hop_length (int): hop length for STFT
        win_length (int): window length for STFT
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()

        self.stft = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=None,
        )

        self.pre_conv = weight_norm(
            nn.Conv2d(2, 32, kernel_size=(3, 9), padding=(1, 4))
        )

        channels = [32, 64, 128, 256]
        self.conv_blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            conv_block = nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                nn.LeakyReLU(negative_slope=LRELU_SLOPE),
            )
            self.conv_blocks.append(conv_block)

        self.post_conv = weight_norm(
            nn.Conv2d(256, 1, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the discriminator.

        Args:
            x (Tensor): waveform input tensor, shape (B, 1, T)

        Returns:
            Tensor: latent output tensor, shape (B, T')
        """

        x = self.stft(x.squeeze(1))
        x = x[:, :-1, :-1]

        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2)

        x = x.contiguous()
        x = self.pre_conv(x)

        for block in self.conv_blocks:
            x = block(x)

        x = self.post_conv(x)
        x = x.flatten(start_dim=1)

        return x


class MultiSpectralDiscriminator(nn.Module):
    r"""Multiple of Spectral STFT Discriminators.

    Args:
        n_ffts (List[int] | None): List of FFT sizes for each sub-discriminator
        hop_lengths (List[int] | None): List of hop lengths for each sub-discriminator
        win_lengths (List[int] | None): List of window lengths for each sub-discriminator
    """

    def __init__(
        self,
        n_ffts: List[int] | None = None,
        hop_lengths: List[int] | None = None,
        win_lengths: List[int] | None = None,
    ):
        super().__init__()

        assert len(n_ffts) == len(hop_lengths) == len(win_lengths), (
            "All parameter lists must have the same length"
        )

        self.sub_discriminators = nn.ModuleList()
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.sub_discriminators.append(
                SpectralDiscriminator(
                    n_fft=n_fft, hop_length=hop_length, win_length=win_length
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Multiple Spectral STFT Discriminator.

        Args:
            x (Tensor): Input waveform, shape (B, 1, T)

        Returns:
            List[torch.Tensor]: List of final logits from each sub-discriminator
        """

        logits = [sub_disc(x) for sub_disc in self.sub_discriminators]
        logits = torch.cat(logits, dim=-1).flatten(start_dim=1)

        return logits


class _SpectroStreamBlock(nn.Module):
    """A single block of the SpectroStream discriminator."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: tuple[int, int]
    ) -> None:
        super().__init__()

        # Single convolution for discriminator efficiency
        kernel_size = (max(3, 2 * stride[0] + 1), max(3, 2 * stride[1] + 1))
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = weight_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor after convolution and activation
        """

        x = self.conv(x)
        x = F.leaky_relu(x, negative_slope=LRELU_SLOPE)

        return x


class SpectroStreamDiscriminator(nn.Module):
    """Single-scale SpectroStream discriminator operating on time-frequency spectrograms."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()

        # STFT transform
        self.stft = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=None,  # Keep complex values
        )

        # Base channel count
        base_channels = 32

        # Initial convolution for 2-channel input (real, imag)
        self.init_conv = weight_norm(
            nn.Conv2d(2, base_channels, kernel_size=(7, 7), padding=(3, 3))
        )

        # Encoder blocks with pre-computed LayerNorm shapes
        channels = [
            base_channels,
            2 * base_channels,
            4 * base_channels,
            8 * base_channels,
            16 * base_channels,
        ]
        strides = [(1, 2), (2, 2), (1, 2), (2, 2)]

        self.encoder_blocks = nn.ModuleList()
        for i, stride in enumerate(strides):
            block = _SpectroStreamBlock(channels[i], channels[i + 1], stride=stride)
            self.encoder_blocks.append(block)

        # Final frequency-wise pooling
        freq_bins = n_fft // 2
        pool_size = max(1, freq_bins // 16)

        self.final_conv = weight_norm(
            nn.Conv2d(
                channels[-1], 1, kernel_size=(1, pool_size), stride=(1, pool_size)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator.

        Args:
            x (Tensor): Input waveform, shape (B, 1, T)

        Returns:
            torch.Tensor: Discriminator logits
        """

        # Convert waveform to spectrogram
        x_single = x.squeeze(1) if x.dim() == 3 else x
        spec = self.stft(x_single)[:, :-1, :-1]  # Remove last freq/time bin

        # Create 2-channel input: real, imaginary
        x = torch.view_as_real(spec)  # (B, freq, time, 2)
        x = x.permute(0, 3, 2, 1)  # (B, 2, time, freq)

        # Initial convolution
        x = x.contiguous()
        x = self.init_conv(x)

        # Apply encoder blocks sequentially
        for block in self.encoder_blocks:
            x = block(x)

        # Apply final frequency pooling
        x = self.final_conv(x)
        logits = x.flatten(start_dim=1)

        return logits


class MultiSpectroStreamDiscriminator(nn.Module):
    """Multi-scale SpectroStream discriminator with configurable STFT parameters.

    Args:
        n_ffts (List[int] | None): FFT sizes for each sub-discriminator
        hop_lengths (List[int] | None): Hop lengths for each sub-discriminator
        win_lengths (List[int] | None): Window lengths for each sub-discriminator
    """

    def __init__(
        self,
        n_ffts: List[int] | None = None,
        hop_lengths: List[int] | None = None,
        win_lengths: List[int] | None = None,
    ):
        super().__init__()

        assert len(n_ffts) == len(hop_lengths) == len(win_lengths), (
            "All parameter lists must have the same length"
        )

        self.sub_discriminators = nn.ModuleList()
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.sub_discriminators.append(
                SpectroStreamDiscriminator(
                    n_fft=n_fft, hop_length=hop_length, win_length=win_length
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale discriminator.

        Args:
            x (Tensor): Input waveform, shape (B, C, T)

        Returns:
            torch.Tensor: Concatenated logits from all discriminator scales
        """
        logits = [sub_disc(x) for sub_disc in self.sub_discriminators]
        logits = torch.cat(logits, dim=-1).flatten(start_dim=1)
        return logits
