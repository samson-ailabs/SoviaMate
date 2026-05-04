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
SEGMENT_SIZE = 8192


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


class BandDiscriminator(nn.Module):
    """Single-band discriminator on a PQMF-decomposed waveform.

    Splits the input into ``period`` equally-spaced subbands via PQMF, then
    runs a 2D conv stack treating subbands as the channel-like axis.

    Args:
        period: Number of PQMF subbands. ``1`` disables PQMF (full waveform).
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
                weight_norm(nn.Conv2d(1, 32, (1, 5), (1, 3), (0, 2))),
                weight_norm(nn.Conv2d(32, 128, (1, 5), (1, 3), (0, 2))),
                weight_norm(nn.Conv2d(128, 512, (1, 5), (1, 3), (0, 2))),
                weight_norm(nn.Conv2d(512, 1024, (1, 5), (1, 3), (0, 2))),
                weight_norm(nn.Conv2d(1024, 1024, (1, 5), (1, 1), (0, 2))),
            ]
        )

        self.proj = weight_norm(nn.Conv2d(1024, 1, (1, 3), (1, 1), (0, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through one PQMF subband stack.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            Discriminator logits of shape (B, N).
        """
        x = self.pqmf(x).unsqueeze(1)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)

        return self.proj(x).flatten(1)


class MultiBandDiscriminator(nn.Module):
    """Stacks BandDiscriminator instances at coprime PQMF subband counts.

    Args:
        periods: PQMF subband counts (coprime values recommended).
        cutoffs: PQMF cutoff frequencies for each period.
        taps: Shared PQMF filter taps.
        beta: Shared PQMF Kaiser window beta parameter.
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
                BandDiscriminator(period=p, taps=taps, cutoff=c, beta=beta)
                for p, c in zip(periods, cutoffs)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through all band discriminators.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            List of logits, one per subband configuration.
        """
        return [disc(x) for disc in self.discriminators]


class TierDiscriminator(nn.Module):
    """Single-tier discriminator on a stratified complex STFT.

    Subsamples the complex STFT along the frequency axis at stride ``stride``
    so the conv stack sees one tier of evenly-interleaved bins spanning DC
    to Nyquist, giving uniform attention across the spectrum.

    Args:
        n_fft: FFT size for STFT.
        win_length: Window length for STFT.
        hop_length: Hop length for STFT.
        stride: Frequency-axis stride defining this tier.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.stride = stride
        self.spec = T.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
        )

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2, 16, (3, 9), (1, 1), (1, 4), (1, 1))),
                weight_norm(nn.Conv2d(16, 32, (3, 9), (1, 2), (1, 4), (1, 1))),
                weight_norm(nn.Conv2d(32, 64, (3, 9), (1, 2), (2, 4), (2, 1))),
                weight_norm(nn.Conv2d(64, 128, (3, 9), (1, 2), (4, 4), (4, 1))),
                weight_norm(nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), (1, 1))),
            ]
        )

        self.proj = weight_norm(nn.Conv2d(256, 1, (3, 3), (1, 1), (1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through one stratified STFT tier.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            Discriminator logits of shape (B, N).
        """
        spec = self.spec(x.squeeze(1))
        spec = spec[:, : -1 : self.stride, :-1]

        x = torch.view_as_real(spec)
        x = x.permute(0, 3, 2, 1)  # (B, 2, T, F)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)

        return self.proj(x).flatten(1)


class MultiTierDiscriminator(nn.Module):
    """Stacks TierDiscriminator instances at multiple STFT resolutions.

    Args:
        n_ffts: FFT sizes per tier.
        win_lengths: Window lengths per tier.
        hop_lengths: Hop lengths per tier.
        strides: Frequency-axis stride per tier.
    """

    def __init__(
        self,
        n_ffts: List[int],
        win_lengths: List[int],
        hop_lengths: List[int],
        strides: List[int],
    ) -> None:
        super().__init__()
        assert len(n_ffts) == len(win_lengths) == len(hop_lengths) == len(strides)

        self.discriminators = nn.ModuleList(
            [
                TierDiscriminator(n_fft=n, win_length=w, hop_length=h, stride=s)
                for n, w, h, s in zip(n_ffts, win_lengths, hop_lengths, strides)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through all stratified STFT tiers.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            List of logits, one per tier.
        """
        return [disc(x) for disc in self.discriminators]


class AudioDiscriminator(nn.Module):
    """Top-level discriminator combining a multi-band PQMF discriminator
    and a multi-tier STFT discriminator.

    Args:
        subband_periods: PQMF subband counts per band sub-discriminator.
        subband_cutoffs: PQMF cutoff frequencies for each subband period.
        n_ffts: FFT sizes per STFT tier.
        win_lengths: Window lengths per STFT tier.
        hop_lengths: Hop lengths per STFT tier.
        tier_strides: Frequency-axis strides for tier stratification.
    """

    def __init__(
        self,
        subband_periods: List[int],
        subband_cutoffs: List[float],
        n_ffts: List[int],
        win_lengths: List[int],
        hop_lengths: List[int],
        tier_strides: List[int],
    ) -> None:
        super().__init__()

        self.band_disc = MultiBandDiscriminator(subband_periods, subband_cutoffs)
        self.tier_disc = MultiTierDiscriminator(
            n_ffts, win_lengths, hop_lengths, tier_strides
        )

    def forward(
        self, fakes: torch.Tensor, reals: torch.Tensor | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Process waveforms through the band and tier discriminators.

        Args:
            fakes: Fake waveforms of shape (B, 1, T).
            reals: Optional real waveforms. If provided, batched with fakes for efficiency.

        Returns:
            Tuple of (fake_logits, real_logits).
            If reals is None, real_logits is an empty list.
        """
        if reals is None:
            fake_logits = self.band_disc(fakes) + self.tier_disc(fakes)
            return fake_logits, []

        combined = torch.cat([fakes, reals], dim=0)
        all_logits = self.band_disc(combined) + self.tier_disc(combined)

        pairs = [logits.chunk(2, dim=0) for logits in all_logits]
        fake_logits, real_logits = zip(*pairs)

        return list(fake_logits), list(real_logits)
