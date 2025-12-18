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


class _DownsampleBlock(nn.Module):
    """Residual block with 2x spatial downsampling and 2x channel expansion."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.channel_ratio = out_channels // in_channels

        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.skip_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2x downsampling with residual connection.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Downsampled tensor of shape (B, 2C, H/2, W/2).
        """
        # Skip path: pool → project → concat
        pooled = self.pool(x)
        skip = torch.cat([pooled, self.skip_conv(pooled)], dim=1)

        # Residual path: activate → conv + pool → activate → project
        h = F.leaky_relu(x, negative_slope=LRELU_SLOPE)
        h = self.conv1(h) + self.pool(h).repeat_interleave(self.channel_ratio, dim=1)
        residual = self.conv2(F.leaky_relu(h, negative_slope=LRELU_SLOPE))

        return F.normalize(skip + 0.4 * residual, p=2, dim=1)


class _UpsampleBlock(nn.Module):
    """Residual block with 2x spatial upsampling and channel reduction."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.channel_ratio = in_channels // out_channels

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2x upsampling with residual connection.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Upsampled tensor of shape (B, C/2, 2H, 2W).
        """
        # Skip path: project → upsample
        skip = self.upsample(self.skip_conv(x))

        # Residual path: activate → conv + upsample → activate → project
        h = F.leaky_relu(x, negative_slope=LRELU_SLOPE)
        h = self.conv1(h) + self.upsample(h)[:, :: self.channel_ratio, :, :]
        residual = self.conv2(F.leaky_relu(h, negative_slope=LRELU_SLOPE))

        return F.normalize(skip + 0.4 * residual, p=2, dim=1)


class SpectralPyramidDiscriminator(nn.Module):
    """Single-scale U-Net discriminator for spectral analysis.

    Uses U-Net encoder-decoder with skip connections to capture features
    at a specific frequency resolution.

    Args:
        win_length: Window length for STFT
        hop_length: Hop length for STFT
    """

    def __init__(self, win_length: int, hop_length: int) -> None:
        super().__init__()

        self.stft = T.Spectrogram(
            n_fft=win_length, win_length=win_length, hop_length=hop_length, power=None
        )

        # Input/output projections
        self.input_conv = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Encoder (downsampling path): 64 → 128 → 256 → 512
        self.down1 = _DownsampleBlock(64, 128)
        self.down2 = _DownsampleBlock(128, 256)
        self.down3 = _DownsampleBlock(256, 512)

        # Skip connection projections (reduce concatenated channels)
        self.skip_proj1 = nn.Conv2d(512, 256, kernel_size=1)
        self.skip_proj2 = nn.Conv2d(256, 128, kernel_size=1)
        self.skip_proj3 = nn.Conv2d(128, 64, kernel_size=1)

        # Decoder (upsampling path): 512 → 256 → 128 → 64
        self.up1 = _UpsampleBlock(512, 256)
        self.up2 = _UpsampleBlock(256, 128)
        self.up3 = _UpsampleBlock(128, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process waveform through spectral U-Net.

        Args:
            x (Tensor): Input waveform of shape (B, 1, T).

        Returns:
            Tensor: Per-pixel logits, shape (B, N).
        """
        spec = self.stft(x.squeeze(1))[:, :-1, :-1]

        h = torch.view_as_real(spec).permute(0, 3, 1, 2)
        e0 = self.input_conv(h.contiguous())

        # Encoder
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)

        # Decoder with skip connections
        d1 = self.skip_proj1(torch.cat([e2, self.up1(e3)], dim=1))
        d2 = self.skip_proj2(torch.cat([e1, self.up2(d1)], dim=1))
        d3 = self.skip_proj3(torch.cat([e0, self.up3(d2)], dim=1))

        return self.output_conv(d3).flatten(start_dim=1)


class MultiSpectralPyramidDiscriminator(nn.Module):
    """Multi-scale spectral pyramid discriminator.

    Each scale has its own U-Net discriminator with independent weights.

    Args:
        win_lengths: Window lengths for each scale
        hop_lengths: Hop lengths for each scale
    """

    def __init__(self, win_lengths: List[int], hop_lengths: List[int]) -> None:
        super().__init__()
        assert len(win_lengths) == len(hop_lengths)

        self.discriminators = nn.ModuleList(
            [
                SpectralPyramidDiscriminator(w, h)
                for w, h in zip(win_lengths, hop_lengths)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process waveform through multi-scale discriminators.

        Args:
            x (Tensor): Input waveform of shape (B, 1, T).

        Returns:
            Tensor: Concatenated logits from all scales, shape (B, N).
        """
        logits = [disc(x) for disc in self.discriminators]
        return torch.cat(logits, dim=1)


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

    BASE_CHANNELS = 32

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


class MultiSpectroStreamDiscriminator(nn.Module):
    """Multi-scale STFT-based discriminator.

    Combines multiple SpectroStreamDiscriminators at different STFT resolutions
    to capture features across various time-frequency trade-offs.

    Args:
        n_ffts: FFT sizes for each scale.
        hop_lengths: Hop lengths for each scale. Defaults to n_fft // 2 (2x overlap).
        win_lengths: Window lengths for each scale. Defaults to n_fft.
    """

    def __init__(
        self,
        n_ffts: List[int],
        hop_lengths: List[int] | None = None,
        win_lengths: List[int] | None = None,
    ) -> None:
        super().__init__()

        if hop_lengths is None:
            hop_lengths = [n // 2 for n in n_ffts]
        if win_lengths is None:
            win_lengths = list(n_ffts)

        self.discriminators = nn.ModuleList(
            [
                SpectroStreamDiscriminator(fft, hop, win)
                for fft, hop, win in zip(n_ffts, hop_lengths, win_lengths)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Process waveform through all scales.

        Args:
            x: Input waveform of shape (B, 1, T) or (B, T).

        Returns:
            List of logits from each scale.
        """
        return [disc(x) for disc in self.discriminators]


class PQMF(nn.Module):
    """Pseudo Quadrature Mirror Filter bank for subband analysis.

    Args:
        period: Number of subbands / downsampling period.
        taps: Number of filter taps (must be even).
        cutoff: Cutoff frequency ratio for prototype filter.
        beta: Kaiser window shape parameter.
    """

    def __init__(
        self, period: int = 4, taps: int = 62, cutoff: float = 0.142, beta: float = 9.0
    ) -> None:
        super().__init__()

        self.period = period
        self.taps = taps

        filters = self.build_filters(period, taps, cutoff, beta)
        self.register_buffer("filters", filters)

    @staticmethod
    def build_filters(
        period: int, taps: int, cutoff: float, beta: float
    ) -> torch.Tensor:
        """Build cosine-modulated filter bank from Kaiser-windowed sinc prototype.

        Args:
            period: Number of subbands / downsampling period.
            taps: Number of filter taps.
            cutoff: Cutoff frequency ratio.
            beta: Kaiser window shape parameter.

        Returns:
            Filter bank tensor of shape (period, 1, taps + 1).
        """
        t = torch.arange(taps + 1, dtype=torch.float32) - taps / 2
        proto = torch.sinc(cutoff * t) * cutoff
        proto = proto * torch.windows.kaiser(taps + 1, beta=beta, sym=True)

        k = torch.arange(period, dtype=torch.float32)
        phase = torch.outer(2 * k + 1, t) * (torch.pi / (2 * period))
        phase = phase + ((-1.0) ** k * torch.pi / 4).unsqueeze(1)

        return (2 * period**0.5 * proto * torch.cos(phase)).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose waveform into subbands.

        Args:
            x: Input waveform of shape (B, 1, T) or (B, T).

        Returns:
            Subband signals of shape (B, period, T // period).
        """
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = F.conv1d(x, self.filters, stride=self.period, padding=self.taps // 2)
        return x


class DiscriminatorP(nn.Module):
    """Period discriminator with PQMF subband decomposition.

    Uses PQMF to decompose waveform into subbands, then applies 2D convolutions
    along the time dimension. Based on BigVGAN's DiscriminatorP architecture.

    Args:
        period: Number of subbands for PQMF decomposition.
        taps: PQMF filter taps.
        cutoff: PQMF cutoff frequency ratio.
        beta: PQMF Kaiser window beta parameter.
    """

    def __init__(
        self, period: int = 4, taps: int = 256, cutoff: float = 0.142, beta: float = 8.0
    ) -> None:
        super().__init__()

        self.pqmf = nn.Identity() if period == 1 else PQMF(period, taps, cutoff, beta)

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(1024, 1024, (5, 1), (1, 1), (2, 0))),
            ]
        )

        self.proj = weight_norm(nn.Conv2d(1024, 1, (3, 1), (1, 1), (1, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through period discriminator.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            Logits tensor of shape (B, N).
        """
        x = self.pqmf(x).transpose(1, 2).unsqueeze(1)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)

        return self.proj(x).flatten(1)


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator combining multiple DiscriminatorP instances.

    Args:
        periods: List of periods (coprime values recommended).
        cutoffs: PQMF cutoff frequencies for each period.
        taps: PQMF filter taps.
        beta: PQMF Kaiser window beta parameter.
    """

    def __init__(
        self,
        periods: List[int],
        cutoffs: List[float],
        taps: int = 256,
        beta: float = 8.0,
    ) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(period=p, taps=taps, cutoff=c, beta=beta)
                for p, c in zip(periods, cutoffs)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through all period discriminators.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            List of logits from each discriminator.
        """
        return [disc(x) for disc in self.discriminators]


class DiscriminatorR(nn.Module):
    """Resolution discriminator operating on magnitude spectrogram.

    Args:
        n_fft: FFT size for STFT.
        win_length: Window length for STFT.
        hop_length: Hop length for STFT.
    """

    def __init__(
        self, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 256
    ) -> None:
        super().__init__()

        self.spec = T.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1
        )

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 16, (3, 9), (1, 1), (1, 4))),
                weight_norm(nn.Conv2d(16, 32, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(32, 64, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(64, 128, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))),
            ]
        )

        self.proj = weight_norm(nn.Conv2d(256, 1, (3, 3), (1, 1), (1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through resolution discriminator.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            Logits tensor of shape (B, N).
        """
        x = self.spec(x.squeeze(1)).unsqueeze(1)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)

        return self.proj(x).flatten(1)


class MultiResolutionDiscriminator(nn.Module):
    """Multi-resolution discriminator combining multiple DiscriminatorR instances.

    Args:
        n_ffts: FFT sizes for each resolution.
        hop_lengths: Hop lengths for each resolution.
        win_lengths: Window lengths for each resolution.
    """

    def __init__(
        self, n_ffts: List[int], win_lengths: List[int], hop_lengths: List[int]
    ) -> None:
        super().__init__()
        assert len(n_ffts) == len(win_lengths) == len(hop_lengths)

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(n_fft=n, win_length=w, hop_length=h)
                for n, w, h in zip(n_ffts, win_lengths, hop_lengths)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through all resolution discriminators.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            List of logits from each discriminator.
        """
        return [disc(x) for disc in self.discriminators]


class MultiScaleDiscriminators(nn.Module):
    """Multi-scale discriminators combining MPD and MRD.

    Combines MultiPeriodDiscriminator (time-domain) and MultiResolutionDiscriminator
    (frequency-domain) for comprehensive audio quality assessment.

    Args:
        periods: Periods for MultiPeriodDiscriminator.
        cutoffs: PQMF cutoff frequencies for each period.
        n_ffts: FFT sizes for MultiResolutionDiscriminator.
        win_lengths: Window lengths for MultiResolutionDiscriminator.
        hop_lengths: Hop lengths for MultiResolutionDiscriminator.
    """

    def __init__(
        self,
        periods: List[int],
        cutoffs: List[float],
        n_ffts: List[int],
        win_lengths: List[int],
        hop_lengths: List[int],
    ) -> None:
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods=periods, cutoffs=cutoffs)
        self.mrd = MultiResolutionDiscriminator(
            n_ffts=n_ffts, win_lengths=win_lengths, hop_lengths=hop_lengths
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Process waveform through all discriminators.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            List of logits from MPD and MRD sub-discriminators.
        """
        return self.mpd(x) + self.mrd(x)
