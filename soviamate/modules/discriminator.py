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


class DiscriminatorP(nn.Module):
    """PQMF-based period discriminator for subband analysis.

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
                weight_norm(nn.Conv2d(1, 32, (1, 5), (1, 3), (0, 2))),
                weight_norm(nn.Conv2d(32, 128, (1, 5), (1, 3), (0, 2))),
                weight_norm(nn.Conv2d(128, 512, (1, 5), (1, 3), (0, 2))),
                weight_norm(nn.Conv2d(512, 1024, (1, 5), (1, 3), (0, 2))),
                weight_norm(nn.Conv2d(1024, 1024, (1, 5), (1, 1), (0, 2))),
            ]
        )

        self.proj = weight_norm(nn.Conv2d(1024, 1, (1, 3), (1, 1), (0, 1)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through filter bank discriminator.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            Tuple of (logits, features) where:
                - logits: Discriminator logits of shape (B, N).
                - features: List of intermediate feature maps from each conv layer.
        """
        x = self.pqmf(x).unsqueeze(1)

        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            features.append(x)

        logits = self.proj(x).flatten(1)
        return logits, features


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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward pass through all period discriminators.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            Tuple of (logits_list, features_list) where:
                - logits_list: List of logits from each discriminator.
                - features_list: List of feature lists from each discriminator.
        """
        logits_list = []
        features_list = []

        for disc in self.discriminators:
            logits, features = disc(x)
            logits_list.append(logits)
            features_list.append(features)

        return logits_list, features_list


class DiscriminatorR(nn.Module):
    """Resolution discriminator operating on complex STFT spectrogram.

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through resolution discriminator.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            Tuple of (logits, features) where:
                - logits: Discriminator logits of shape (B, N).
                - features: List of intermediate feature maps from each conv layer.
        """
        spec = self.spec(x.squeeze(1))
        spec = spec[:, :-1, :-1]  # Omit Nyquist and last frame

        x = torch.view_as_real(spec)
        x = x.permute(0, 3, 2, 1)  # (B, 2, T, F)

        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            features.append(x)

        logits = self.proj(x).flatten(1)
        return logits, features


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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward pass through all resolution discriminators.

        Args:
            x: Input waveform of shape (B, 1, T).

        Returns:
            Tuple of (logits_list, features_list) where:
                - logits_list: List of logits from each discriminator.
                - features_list: List of feature lists from each discriminator.
        """
        logits_list = []
        features_list = []

        for disc in self.discriminators:
            logits, features = disc(x)
            logits_list.append(logits)
            features_list.append(features)

        return logits_list, features_list


class MultiScaleDiscriminators(nn.Module):
    """Multi-scale discriminators combining MPD and MS-STFT style MRD.

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

    def forward(
        self, fakes: torch.Tensor, reals: torch.Tensor | None = None
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        """Process waveforms through MPD and MRD discriminators.

        Args:
            fakes: Fake waveforms of shape (B, 1, T).
            reals: Optional real waveforms. If provided, batches with fakes for efficiency.

        Returns:
            Tuple of (fake_logits, real_logits, fake_features, real_features).
            If reals is None, real_logits and real_features are empty lists.
        """
        # Single input: only fakes
        if reals is None:
            mpd_logits, mpd_features = self.mpd(fakes)
            mrd_logits, mrd_features = self.mrd(fakes)

            fake_logits = mpd_logits + mrd_logits
            fake_features = mpd_features + mrd_features

            return fake_logits, [], fake_features, []

        # Batched: fakes + reals
        combined = torch.cat([fakes, reals], dim=0)

        mpd_logits, mpd_features = self.mpd(combined)
        mrd_logits, mrd_features = self.mrd(combined)

        # Combine MPD and MRD outputs
        all_logits = mpd_logits + mrd_logits
        all_features = mpd_features + mrd_features

        # Split into fake and real
        fake_logits, real_logits = [], []
        fake_features, real_features = [], []

        for logits in all_logits:
            f_logits, r_logits = logits.chunk(2, dim=0)
            fake_logits.append(f_logits)
            real_logits.append(r_logits)

        for features in all_features:
            f_feats = [feat.chunk(2, dim=0)[0] for feat in features]
            r_feats = [feat.chunk(2, dim=0)[1] for feat in features]
            fake_features.append(f_feats)
            real_features.append(r_feats)

        return fake_logits, real_logits, fake_features, real_features
