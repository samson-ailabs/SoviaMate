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

"""Audio processing modules for extracting spectrograms and reconstructing waveforms"""

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as TAudio
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class _FeedForwardNetwork(nn.Module):
    """Feedforward network with residual connections.

    Args:
        feature_dim (int): Input and output feature dimension.
        num_blocks (int, optional): Number of blocks. Defaults to 1.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """

    def __init__(self, feature_dim: int, num_blocks: int = 1, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, 2 * feature_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * feature_dim, feature_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feedforward transformations with residual connections.

        Args:
            x (Tensor): Input tensor of shape (B, T, C) where B is batch size,
                T is sequence length, and C is channel dimension.

        Returns:
            Tensor: Output tensor of shape (B, T, C).
        """
        for block in self.blocks:
            x = x + block(x)
        return x


class SpectrogramProcessor(nn.Module):
    """STFT-based spectrogram processor with magnitude and phase streams.

    Args:
        frame_stacking (int): Number of frames to stack for downsampling.
        window_length (int): Window length for STFT (n_fft).
        hop_length (int): Hop length for STFT.
        output_dim (int): Output dimension after final projection.
        mag_dim (int, optional): Magnitude stream dimension. Defaults to 512.
        phase_dim (int, optional): Phase stream dimension. Defaults to 256.
        phase_grad_dim (int, optional): Phase gradient stream dimension. Defaults to 256.
        num_ffn_blocks (int, optional): Number of feedforward blocks per stream. Defaults to 2.
    """

    def __init__(
        self,
        frame_stacking: int,
        window_length: int,
        hop_length: int,
        output_dim: int,
        mag_dim: int = 512,
        phase_dim: int = 256,
        phase_grad_dim: int = 256,
        num_ffn_blocks: int = 2,
    ):
        super().__init__()

        self.frame_stacking = frame_stacking
        self.window_length = window_length
        self.hop_length = hop_length
        self.mag_dim = mag_dim
        self.phase_dim = phase_dim
        self.phase_grad_dim = phase_grad_dim

        self.specgram = TAudio.Spectrogram(
            n_fft=window_length,
            win_length=window_length,
            hop_length=hop_length,
            power=None,
        )

        n_bins = window_length // 2 + 1
        stacked_bins = n_bins * frame_stacking

        # Separate embeddings for each stream
        self.mag_embed = nn.Linear(stacked_bins, mag_dim)
        self.phase_embed = nn.Linear(stacked_bins, phase_dim)
        self.phase_grad_embed = nn.Linear(stacked_bins, phase_grad_dim)

        # Feedforward networks with residual connections
        self.mag_ffn = _FeedForwardNetwork(mag_dim, num_ffn_blocks)
        self.phase_ffn = _FeedForwardNetwork(phase_dim, num_ffn_blocks)
        self.phase_grad_ffn = _FeedForwardNetwork(phase_grad_dim, num_ffn_blocks)

        # Final projection to output dimension
        total_dim = mag_dim + phase_dim + phase_grad_dim
        self.projector = nn.Linear(total_dim, output_dim)

    def _compute_phase_gradient(self, phase: torch.Tensor) -> torch.Tensor:
        """Compute temporal gradient of phase with 2π discontinuity correction.

        Calculates the rate of phase change over time, removing artificial jumps
        caused by phase wrapping. This provides explicit phase velocity information
        that helps the network model phase dynamics more effectively.

        Args:
            phase (Tensor): Phase in [-π, π] with shape (B, T, n_bins).

        Returns:
            Tensor: Phase temporal gradient (B, T, n_bins).
        """
        # Compute phase difference between consecutive frames
        phase_diff = torch.diff(phase, dim=1)

        # Wrap differences to [-π, π] to remove 2π jumps
        phase_diff_wrapped = torch.angle(torch.exp(1j * phase_diff))

        # Prepend zero for first frame (no gradient at t=0)
        gradient = torch.cat(
            [torch.zeros_like(phase[:, :1, :]), phase_diff_wrapped], dim=1
        )

        return gradient

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass extracting magnitude and phase gradient features.

        Args:
            waveforms (Tensor): Input tensor with shape (B, T, 1).
            lengths (Tensor): Audio sample lengths with shape (B,).

        Returns:
            Tensor: Output features with shape (B, T // (hop_length * frame_stacking), D).
            Tensor: Feature sequence lengths with shape (B,).
        """
        if waveforms.size(2) != 1:
            raise ValueError("The audio signal should be mono-channel.")

        # Extract complex spectrograms: (B, n_bins, T') -> (B, T', n_bins)
        specs = self.specgram(waveforms.squeeze(2)).transpose(1, 2)
        batch_size, num_frames, n_bins = specs.shape

        # Stream 1: Log-magnitude
        log_mag = torch.log(torch.clamp(specs.abs(), min=1e-5))

        # Stream 2: Raw phase (wrapped)
        phase = specs.angle()

        # Stream 3: Unwrapped phase temporal gradient
        phase_grad = self._compute_phase_gradient(phase)

        # Pad to make divisible by frame_stacking
        remainder = num_frames % self.frame_stacking
        if remainder != 0:
            pad_frames = self.frame_stacking - remainder
            log_mag = F.pad(log_mag, (0, 0, 0, pad_frames))
            phase = F.pad(phase, (0, 0, 0, pad_frames))
            phase_grad = F.pad(phase_grad, (0, 0, 0, pad_frames))

        # Stack frames for each stream: (B, T' // stack, stack * n_bins)
        stacked_frames = log_mag.size(1) // self.frame_stacking
        log_mag_stacked = log_mag.reshape(
            batch_size, stacked_frames, self.frame_stacking * n_bins
        )
        phase_stacked = phase.reshape(
            batch_size, stacked_frames, self.frame_stacking * n_bins
        )
        phase_grad_stacked = phase_grad.reshape(
            batch_size, stacked_frames, self.frame_stacking * n_bins
        )

        # Embed each stream to its target dimension
        mag_features = self.mag_embed(log_mag_stacked)
        phase_features = self.phase_embed(phase_stacked)
        phase_grad_features = self.phase_grad_embed(phase_grad_stacked)

        # Process each stream through feedforward networks
        mag_features = self.mag_ffn(mag_features)
        phase_features = self.phase_ffn(phase_features)
        phase_grad_features = self.phase_grad_ffn(phase_grad_features)

        # Apply final projection to desired output dimension
        features = torch.cat((mag_features, phase_features, phase_grad_features), dim=2)
        features = self.projector(features)

        # Calculate per-sample output lengths matching actual STFT output
        spec_lengths = 1 + lengths.div(self.hop_length, rounding_mode="floor")
        output_lengths = torch.ceil(spec_lengths / self.frame_stacking)

        return features, output_lengths.to(lengths.dtype)


class InverseSpectrogramProcessor(nn.Module):
    """iSTFT-based inverse processor for waveform reconstruction.

    Args:
        frame_stacking (int): Number of frames to unstack for upsampling.
        window_length (int): Window length for iSTFT (n_fft).
        hop_length (int): Hop length for iSTFT.
        input_dim (int): Input feature dimension.
        mag_dim (int, optional): Magnitude branch dimension. Defaults to 512.
        phase_dim (int, optional): Phase branch dimension. Defaults to 512.
        num_ffn_blocks (int, optional): Number of feedforward blocks per branch. Defaults to 2.
    """

    def __init__(
        self,
        frame_stacking: int,
        window_length: int,
        hop_length: int,
        input_dim: int,
        mag_dim: int = 512,
        phase_dim: int = 512,
        num_ffn_blocks: int = 2,
    ):
        super().__init__()

        self.frame_stacking = frame_stacking
        self.window_length = window_length
        self.hop_length = hop_length
        self.mag_dim = mag_dim
        self.phase_dim = phase_dim

        self.inverse_specgram = TAudio.InverseSpectrogram(
            n_fft=window_length,
            win_length=window_length,
            hop_length=hop_length,
        )

        n_bins = window_length // 2 + 1
        stacked_bins = n_bins * frame_stacking

        # Magnitude branch: predicts log-magnitude
        self.mag_proj_in = nn.Linear(input_dim, mag_dim)
        self.mag_ffn = _FeedForwardNetwork(mag_dim, num_ffn_blocks)
        self.mag_proj_out = nn.Linear(mag_dim, stacked_bins)

        # Phase branch: predicts real and imaginary components
        self.phase_proj_in = nn.Linear(input_dim, phase_dim)
        self.phase_ffn = _FeedForwardNetwork(phase_dim, num_ffn_blocks)
        self.phase_proj_out = nn.Linear(phase_dim, 2 * stacked_bins)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        max_output_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct waveform from decoder features via inverse STFT.

        Args:
            features: Input tensor with shape (B, T, D).
            lengths: Feature sequence lengths with shape (B,).
            max_output_length: Maximum output audio length for exact reconstruction.

        Returns:
            Tensor: Output waveform tensor with shape (B, max_output_length or inferred, 1).
            Tensor: Output audio lengths with shape (B,).
        """
        batch_size, stacked_frames, _ = features.shape
        n_bins = self.window_length // 2 + 1

        # Magnitude branch: proj_in → FFN → proj_out
        mag_hidden = self.mag_proj_in(features)
        mag_hidden = self.mag_ffn(mag_hidden)
        log_mag = self.mag_proj_out(mag_hidden)

        # Phase branch: proj_in → FFN → proj_out (predicts real and imaginary)
        phase_hidden = self.phase_proj_in(features)
        phase_hidden = self.phase_ffn(phase_hidden)
        phase_output = self.phase_proj_out(phase_hidden)
        real, imag = phase_output.chunk(2, dim=2)

        # Unstack frames: (B, T, stacked_bins) -> (B, T * stack, n_bins)
        unstacked_frames = stacked_frames * self.frame_stacking
        log_mag = log_mag.reshape(batch_size, unstacked_frames, n_bins)
        real = real.reshape(batch_size, unstacked_frames, n_bins)
        imag = imag.reshape(batch_size, unstacked_frames, n_bins)

        # Build complex spectrogram from magnitude and phase
        mag, phase = log_mag.exp(), torch.atan2(imag, real)
        specs = torch.complex(mag * phase.cos(), mag * phase.sin())

        # Convert to waveform via inverse STFT: (B, n_bins, T') -> (B, T_audio)
        waveforms = self.inverse_specgram(
            specs.transpose(1, 2), length=max_output_length
        )

        # Calculate output lengths: unstacked_frames -> audio samples
        spec_lengths = lengths * self.frame_stacking
        output_lengths = (spec_lengths - 1) * self.hop_length + self.window_length

        # Clamp to actual waveform size if max_output_length is provided
        if max_output_length is not None:
            output_lengths = torch.clamp(output_lengths, max=max_output_length)

        return waveforms.unsqueeze(2), output_lengths


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
