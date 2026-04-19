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

"""Audio processing modules for waveform feature extraction and reconstruction."""

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

# Power-compression exponent for the complex streams (|s|^β · cos φ, |s|^β · sin φ).
# β < 1 compresses dynamic range so the linear projection can attend to quiet bins.
POWER_COMPRESSION_BETA: float = 0.3


class SpectralAnalyzer(nn.Module):
    r"""Waveform-to-feature projection via non-overlapping rfft.

    Each sub-frame emits 3 streams per bin — ``log|s|``, ``|s|^β·cos φ``,
    ``|s|^β·sin φ`` — with ``β < 1`` compressing the complex streams.
    Adjacent sub-frames are stacked before projection.

    Args:
        output_dim (int): Projected feature dimension.
        hop_length (int): Samples per sub-frame.
        frame_stacking (int): Sub-frames per feature frame. Default: ``2``.
    """

    def __init__(self, output_dim: int, hop_length: int, frame_stacking: int = 2):
        super().__init__()

        self.hop_length = hop_length
        self.frame_stacking = frame_stacking
        self.fft_bins = hop_length // 2 + 1
        self.sub_frame_dim = self.fft_bins * 3

        self.proj = nn.Linear(self.sub_frame_dim * frame_stacking, output_dim)

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Project waveforms into features.

        Args:
            waveforms (Tensor): Input waveform with shape ``(B, T, 1)``.
            lengths (Tensor): Per-sample waveform lengths with shape ``(B,)``.

        Returns:
            Tuple[Tensor, Tensor]: ``(features, output_lengths)`` with shapes
                ``(B, T', output_dim)`` and ``(B,)``.
        """
        x = waveforms.squeeze(2)
        batch_size = x.size(0)

        stride = self.hop_length * self.frame_stacking
        remainder = x.size(1) % stride
        if remainder != 0:
            x = F.pad(x, (0, stride - remainder))

        sub_frames = x.reshape(batch_size, -1, self.hop_length)
        spectrum = torch.fft.rfft(sub_frames)

        magnitude = spectrum.abs().clamp(min=1e-5)
        log_mag = magnitude.log()

        mag_beta = magnitude.pow(POWER_COMPRESSION_BETA)
        real_c = mag_beta * (spectrum.real / magnitude)
        imag_c = mag_beta * (spectrum.imag / magnitude)

        x = torch.cat([log_mag, real_c, imag_c], dim=-1)
        x = x.reshape(batch_size, x.size(1) // self.frame_stacking, -1)

        features = self.proj(x)
        output_lengths = (lengths + stride - 1).div(stride, rounding_mode="floor")

        return features, output_lengths


class SpectralSynthesizer(nn.Module):
    r"""Feature-to-waveform projection via non-overlapping irfft.

    Predicts 3 streams per bin — ``(log_mag, real, imag)`` — and composes
    ``spec = exp(log_mag) · (real + j·imag)``. Leaving ``(real, imag)``
    un-normalized lets all three channels carry magnitude and phase jointly.

    Args:
        input_dim (int): Input feature dimension.
        hop_length (int): Samples per sub-frame.
        frame_stacking (int): Sub-frames per feature frame. Default: ``2``.
    """

    def __init__(self, input_dim: int, hop_length: int, frame_stacking: int = 2):
        super().__init__()

        self.hop_length = hop_length
        self.frame_stacking = frame_stacking
        self.fft_bins = hop_length // 2 + 1
        self.sub_frame_dim = self.fft_bins * 3

        self.proj = nn.Linear(input_dim, self.sub_frame_dim * frame_stacking)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        max_output_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Reconstruct waveforms from features.

        Args:
            features (Tensor): Input features with shape ``(B, T, D)``.
            lengths (Tensor): Per-sample feature lengths with shape ``(B,)``.
            max_output_length (int, optional): Cap on output sample count.

        Returns:
            Tuple[Tensor, Tensor]: ``(waveforms, output_lengths)`` with shapes
                ``(B, T', 1)`` and ``(B,)``.
        """
        batch_size, num_frames, _ = features.shape

        x = self.proj(features)
        x = x.reshape(batch_size, num_frames * self.frame_stacking, self.sub_frame_dim)
        log_mag, real, imag = x.chunk(3, dim=-1)

        mag_scale = log_mag.exp()
        spectrum = torch.complex(mag_scale * real, mag_scale * imag)

        frames = torch.fft.irfft(spectrum, n=self.hop_length)
        waveforms = frames.reshape(batch_size, -1, 1)
        output_lengths = lengths * self.hop_length * self.frame_stacking

        if max_output_length is not None:
            output_lengths = torch.clamp(output_lengths, max=max_output_length)
            waveforms = waveforms[:, :max_output_length]

        return waveforms, output_lengths


class SpecAugmentProcessor(nn.Module):
    r"""SpecAugment-style masking for hidden representations.

    Applies time and hidden dimension masking to learned features
    in a batched manner without Python for loops over samples.

    Args:
        time_mask_param (float): Ratio of sequence length to mask (e.g., 0.05 = 5%). Default 0.05.
        freq_mask_param (float): Ratio of hidden dimension to mask. Default 0.0 (disabled).
        num_time_masks (int): Number of time masks per sample. Default 10.
        num_freq_masks (int): Number of hidden dimension masks per sample. Default 0 (disabled).
        mask_value (float): Value to fill masked regions. Default 0.0.
    """

    def __init__(
        self,
        time_mask_param: float = 0.05,
        freq_mask_param: float = 0.0,
        num_time_masks: int = 5,
        num_freq_masks: int = 0,
        mask_value: float = 0.0,
    ):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.mask_value = mask_value

    def forward(
        self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Apply feature masking to hidden representations.

        Args:
            features (Tensor): Hidden features with shape `(B, T, D)`.
            lengths (Tensor, optional): Actual sequence lengths with shape `(B,)`.
                If None, all sequences assumed to have length T.

        Returns:
            Tensor: Masked features with shape `(B, T, D)`.
        """
        batch_size, time_steps, _ = features.shape
        device = features.device

        # Set default lengths if not provided
        if lengths is None:
            lengths = torch.full(
                (batch_size,), time_steps, dtype=torch.long, device=device
            )

        # Apply feature masking (vectorized)
        features = self._apply_freq_mask(features)

        # Apply time masking (vectorized)
        features = self._apply_time_mask(features, lengths)

        return features

    def _apply_freq_mask(self, features: torch.Tensor) -> torch.Tensor:
        r"""Apply hidden dimension masking in a vectorized manner.

        Args:
            features (Tensor): Input tensor with shape `(B, T, D)`.

        Returns:
            Tensor: Features with hidden dimension masks applied.
        """
        batch_size, _, hidden_dim = features.shape
        device = features.device

        for _ in range(self.num_freq_masks):
            # Calculate max mask width as ratio of hidden dimension
            max_mask_width = max(1, int(hidden_dim * self.freq_mask_param))

            # Generate random mask widths for all samples (batch_size,)
            mask_width = (torch.rand(batch_size, device=device) * max_mask_width).long()

            # Generate random start positions ensuring start + width <= hidden_dim
            max_start = (hidden_dim - mask_width).clamp(min=0)
            mask_start = (torch.rand(batch_size, device=device) * max_start).long()

            # Create feature mask using broadcasting
            feat_indices = torch.arange(hidden_dim, device=device)[None, None, :]

            # Mask condition: mask_start <= feat_idx < mask_start + mask_width
            mask = (feat_indices >= mask_start[:, None, None]) & (
                feat_indices < (mask_start + mask_width)[:, None, None]
            )

            # Apply mask (vectorized across entire batch)
            features = torch.where(mask, self.mask_value, features)

        return features

    def _apply_time_mask(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        r"""Apply time masking in a vectorized manner.

        Args:
            features (Tensor): Input tensor with shape `(B, T, D)`.
            lengths (Tensor): Actual sequence lengths with shape `(B,)`.

        Returns:
            Tensor: Features with time masks applied.
        """
        batch_size, time_steps, _ = features.shape
        device = features.device

        for _ in range(self.num_time_masks):
            # Calculate max mask width as ratio of each sequence length
            max_mask_width = (lengths * self.time_mask_param).long().clamp(min=1)

            # Generate random mask widths (batch_size,)
            mask_width = (torch.rand(batch_size, device=device) * max_mask_width).long()

            # Generate random start positions (batch_size,)
            max_start = (lengths - mask_width).clamp(min=0)
            mask_start = (torch.rand(batch_size, device=device) * max_start).long()

            # Create time mask using broadcasting
            time_indices = torch.arange(time_steps, device=device)[None, :, None]

            # Mask condition: mask_start <= time_idx < mask_start + mask_width
            mask = (time_indices >= mask_start[:, None, None]) & (
                time_indices < (mask_start + mask_width)[:, None, None]
            )

            # Apply mask (vectorized across entire batch)
            features = torch.where(mask, self.mask_value, features)

        return features


class SpliceOutProcessor(nn.Module):
    r"""SpliceOut Audio Augmentation for ASR Decoder Training.

    Removes contiguous time-step segments from feature sequences and concatenates
    the remaining parts. More efficient than masking-based augmentation.

    Args:
        num_splices (int): Number of splices to remove. Default: 2.
        max_splice_length (int): Maximum splice length in frames. Default: 10.
        probability (float): Probability of applying augmentation. Default: 1.0 (always apply).
    """

    def __init__(
        self,
        num_splices: int = 2,
        max_splice_length: int = 10,
        probability: float = 1.0,
    ):
        super().__init__()
        self.num_splices = num_splices
        self.max_splice_length = max_splice_length
        self.probability = probability

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Apply SpliceOut augmentation.

        Args:
            features (Tensor): Input features with shape `(B, T, D)`.
            lengths (Tensor): Actual sequence lengths with shape `(B,)`.

        Returns:
            Tuple of (spliced_features, new_lengths).
        """
        batch_size, max_len, feature_dim = features.shape
        device = features.device

        if self.num_splices == 0 or random.random() > self.probability:
            return features, lengths

        # Generate random splice lengths for each splice
        splice_lengths = torch.randint(
            0, self.max_splice_length, (batch_size, self.num_splices), device=device
        )

        # Calculate valid range for splice start positions
        max_starts = torch.clamp(lengths.unsqueeze(1) - splice_lengths, min=0)

        # Randomly sample start positions within valid range
        rand_vals = torch.rand((batch_size, self.num_splices), device=device)
        splice_starts = torch.clamp(
            (rand_vals * (max_starts + 1)).long(), max=max_starts
        )

        # Calculate splice end positions
        splice_ends = torch.clamp(
            splice_starts + splice_lengths, max=lengths.unsqueeze(1)
        )

        # Create removal mask via 3D broadcasting (1, T, 1) × (B, 1, N) → (B, T, N)
        time_idx = torch.arange(max_len, device=device).view(1, -1, 1)
        starts_expanded = splice_starts.unsqueeze(1)
        ends_expanded = splice_ends.unsqueeze(1)
        in_splice = (time_idx >= starts_expanded) & (time_idx < ends_expanded)

        # Mask out padding positions outside sequence boundaries
        is_valid_pos = torch.arange(max_len, device=device) < lengths.unsqueeze(1)
        final_mask = (~in_splice.any(dim=2)) & is_valid_pos

        # Extract kept frames and split by sequence length
        features_flat = features.reshape(-1, feature_dim)
        mask_flat = final_mask.reshape(-1)
        selected_frames = features_flat[mask_flat]

        # Repackage into batch with new lengths and add padding
        new_lengths = final_mask.sum(dim=1)
        spliced_sequences = torch.split(selected_frames, new_lengths.tolist())

        padded_outputs = pad_sequence(
            spliced_sequences, batch_first=True, padding_value=0.0
        )

        return padded_outputs, new_lengths
