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

import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.2
SEGMENT_SIZE = 16384


class _SpectroStreamBlock(nn.Module):
    """Pre-activation residual block with strided downsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for spatial downsampling (time, freq).
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: tuple[int, int]
    ) -> None:
        super().__init__()

        kernel_size = (max(3, 2 * stride[0] + 1), max(3, 2 * stride[1] + 1))
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.main = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.LeakyReLU(negative_slope=LRELU_SLOPE),
            nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1)),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.LeakyReLU(negative_slope=LRELU_SLOPE),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        )

        layers = []
        if stride != (1, 1):
            layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
        if in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)))
        self.skip = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block with skip connection.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C', H', W').
        """
        return self.skip(x) + self.main(x)


class SpectroStreamDiscriminator(nn.Module):
    """Single-scale STFT-based discriminator for mono audio.

    Processes waveform through STFT to create 3-channel spectrogram input
    (real, imaginary, magnitude), then applies residual encoder blocks.

    Args:
        n_fft: FFT size for STFT.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
    """

    BASE_CHANNELS = 16

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        ch = self.BASE_CHANNELS

        self.stft = T.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
        )

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(3, ch, kernel_size=7, padding=3)),
                weight_norm(nn.Conv2d(1 * ch, 2 * ch, (3, 5), (1, 2), (1, 2))),
                weight_norm(nn.Conv2d(2 * ch, 4 * ch, (5, 5), (2, 2), (2, 2))),
                weight_norm(nn.Conv2d(4 * ch, 8 * ch, (3, 5), (1, 2), (1, 2))),
                weight_norm(nn.Conv2d(8 * ch, 16 * ch, (5, 5), (2, 2), (2, 2))),
            ]
        )

        freq_bins = n_fft // 2  # After omitting Nyquist
        final_freq = max(1, freq_bins // 16)

        self.proj = weight_norm(nn.Conv2d(16 * ch, 1, (1, final_freq), (1, final_freq)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SpectroStream discriminator.

        Args:
            x: Input waveform of shape (B, 1, T) or (B, T).

        Returns:
            Discriminator logits of shape (B, num_logits).
        """
        spec = self.stft(x.squeeze(1) if x.dim() == 3 else x)
        spec = spec[:, :-1, :-1]  # Omit Nyquist and last frame

        rim = torch.view_as_real(spec)
        mag = torch.abs(spec).unsqueeze(-1)

        x = torch.cat([rim, mag], dim=-1)
        x = x.permute(0, 3, 2, 1)  # (B, 3, T, F)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)

        return self.proj(x).flatten(1)


class MultiSpectroStreamDiscriminators(nn.Module):
    """Multi-scale SpectroStream discriminators for mono audio.

    Args:
        n_ffts: FFT sizes for each scale.
        win_lengths: Window lengths for each scale. Defaults to n_fft.
        hop_lengths: Hop lengths for each scale. Defaults to n_fft // 2 (2x overlap).
        num_active: Number of discriminators to randomly sample per forward pass.
    """

    def __init__(
        self,
        n_ffts: List[int],
        win_lengths: List[int] | None = None,
        hop_lengths: List[int] | None = None,
        num_active: int | None = None,
    ) -> None:
        super().__init__()

        if hop_lengths is None:
            hop_lengths = [n // 2 for n in n_ffts]
        if win_lengths is None:
            win_lengths = list(n_ffts)

        self.num_active = num_active if num_active is not None else len(n_ffts)

        self.discriminators = nn.ModuleList(
            [
                SpectroStreamDiscriminator(fft, hop, win)
                for fft, hop, win in zip(n_ffts, hop_lengths, win_lengths)
            ]
        )

    def forward(
        self, fakes: torch.Tensor, reals: torch.Tensor | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Process waveforms through randomly sampled scales.

        Args:
            fakes: Fake waveforms of shape (B, 1, T) or (B, T).
            reals: Optional real waveforms. If provided, batches with fakes for efficiency.

        Returns:
            Tuple of (fake_logits, real_logits). If reals is None, real_logits is empty.
        """
        # Select discriminators to use
        num_discs = len(self.discriminators)
        if self.num_active >= num_discs or not self.training:
            discs = self.discriminators
        else:
            idxs = random.sample(range(num_discs), self.num_active)
            discs = [self.discriminators[i] for i in idxs]

        # Single input: only fakes
        if reals is None:
            return [disc(fakes) for disc in discs], []

        # Batched: fakes + reals
        combined = torch.cat([fakes, reals], dim=0)
        outputs = [disc(combined) for disc in discs]

        fake_logits, real_logits = zip(*[out.chunk(2, dim=0) for out in outputs])
        return list(fake_logits), list(real_logits)
