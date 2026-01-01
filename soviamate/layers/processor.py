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

import math
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio.transforms as TAudio
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class SpectrogramProcessor(nn.Module):
    r"""Spectrogram extraction processor with frame stacking.

    Args:
        frame_stacking (int): Number of frames to stack for downsampling.
        window_length (int): Window length for STFT (n_fft).
        hop_length (int): Hop length for STFT.
        output_dim (int): Output dimension after linear projection.
    """

    def __init__(
        self, frame_stacking: int, window_length: int, hop_length: int, output_dim: int
    ):
        super().__init__()
        self.frame_stacking = frame_stacking
        self.window_length = window_length
        self.hop_length = hop_length

        n_bins = window_length // 2 + 1
        stacked_dim = 2 * n_bins * frame_stacking

        self.specgram = TAudio.Spectrogram(
            n_fft=window_length,
            win_length=window_length,
            hop_length=hop_length,
            power=None,
        )
        self.projector = nn.Linear(stacked_dim, output_dim)

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the spectrogram processor.

        Args:
            waveforms (Tensor): Input tensor with shape `(B, T, 1)`.
            lengths (Tensor): Audio sample lengths with shape `(B,)`.

        Returns:
            Tensor: Output features with shape `(B, T // (hop_length * frame_stacking), D)`.
            Tensor: Feature sequence lengths with shape `(B,)`.
        """

        if waveforms.size(2) != 1:
            raise ValueError("The audio signal should be mono-channel.")

        # Extract spectrograms: (B, n_bins, T') -> (B, T', n_bins)
        specs = self.specgram(waveforms.squeeze(2)).transpose(1, 2)
        mag, phase = specs.abs().clamp(1e-9).log(), specs.angle()

        # Concatenate magnitude and phase: (B, T', 2 * n_bins)
        features = torch.cat((mag, phase), dim=2)
        batch_size, num_frames, _ = features.shape

        # Pad to make divisible by frame_stacking
        remainder = num_frames % self.frame_stacking
        if remainder != 0:
            pad_frames = self.frame_stacking - remainder
            features = F.pad(features, (0, 0, 0, pad_frames))

        # Reshape to stack frames: (B, T' // stack, stack * 2 * n_bins)
        stacked_frames = features.size(1) // self.frame_stacking
        features = features.reshape(batch_size, stacked_frames, -1)

        # Calculate output lengths (spectrogram frames / frame_stacking)
        spec_lengths = torch.floor(lengths / self.hop_length) + 1
        output_lengths = torch.ceil(spec_lengths / self.frame_stacking)

        # Apply linear projection to desired output dimension
        features = self.projector(features)

        return features, output_lengths.to(dtype=lengths.dtype)


class InverseSpectrogramProcessor(nn.Module):
    r"""Inverse spectrogram processor for reconstructing waveforms from spectrograms.

    Args:
        frame_stacking (int): Number of frames to unstack for upsampling.
        window_length (int): Window length for iSTFT (n_fft).
        hop_length (int): Hop length for iSTFT.
        input_dim (int): Input dimension before linear projection.
    """

    def __init__(
        self, frame_stacking: int, window_length: int, hop_length: int, input_dim: int
    ):
        super().__init__()
        self.frame_stacking = frame_stacking
        self.window_length = window_length
        self.hop_length = hop_length

        n_bins = window_length // 2 + 1
        stacked_dim = 2 * n_bins * frame_stacking

        self.inverse_specgram = TAudio.InverseSpectrogram(
            n_fft=window_length,
            win_length=window_length,
            hop_length=hop_length,
        )
        self.projector = nn.Linear(input_dim, stacked_dim)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        max_output_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the inverse spectrogram processor.

        Args:
            features (Tensor): Input tensor with shape `(B, T, D)`.
            lengths (Tensor): Feature sequence lengths with shape `(B,)`.
            max_output_length (int, optional): Maximum output audio length for exact reconstruction.

        Returns:
            Tensor: Output waveform tensor with shape `(B, max_output_length or inferred, 1)`.
            Tensor: Output audio lengths with shape `(B,)`.
        """
        batch_size, stacked_frames, _ = features.shape
        n_bins = self.window_length // 2 + 1

        # Apply linear projection: (B, T, D) -> (B, T, 2 * n_bins * frame_stacking)
        features = self.projector(features)

        # Unstack frames: (B, T, stack * 2 * n_bins) -> (B, T * stack, 2 * n_bins)
        features = features.reshape(
            batch_size, stacked_frames * self.frame_stacking, 2 * n_bins
        )

        # Convert back to complex spectrogram
        mag, phase = features.chunk(2, dim=2)
        specs = torch.polar(mag.exp(), phase.float())

        # Convert to waveform: (B, n_bins, T') -> (B, T_audio)
        waveforms = self.inverse_specgram(
            specs.transpose(1, 2), length=max_output_length
        )

        # Calculate output lengths: unstacked_frames -> audio samples
        spec_lengths = lengths * self.frame_stacking
        output_lengths = (spec_lengths - 1) * self.hop_length

        # Clamp to actual waveform size if max_output_length is provided
        if max_output_length is not None:
            output_lengths = torch.clamp(output_lengths, max=max_output_length)

        return waveforms.unsqueeze(2), output_lengths


class WaveformPatcher(nn.Module):
    r"""Converts raw waveform into patch embeddings.

    Args:
        patch_size (int): Number of audio samples per patch.
        output_dim (int): Output embedding dimension after projection.
    """

    def __init__(self, patch_size: int, output_dim: int):
        super().__init__()

        self.patch_size = patch_size
        self.output_dim = output_dim

        self.linear1 = nn.Linear(patch_size, output_dim, bias=False)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=True)

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the waveform patcher.

        Args:
            waveforms (Tensor): Input tensor with shape `(B, T, 1)`.
            lengths (Tensor): Audio sample lengths with shape `(B,)`.

        Returns:
            Tensor: Output features with shape `(B, T // patch_size, D)`.
            Tensor: Feature sequence lengths with shape `(B,)`.
        """
        if waveforms.size(2) != 1:
            raise ValueError("The audio signal should be mono-channel.")

        x = waveforms.squeeze(2)
        batch_size, num_samples = x.shape

        remainder = num_samples % self.patch_size
        if remainder != 0:
            x = F.pad(x, (0, self.patch_size - remainder))

        num_patches = x.size(1) // self.patch_size
        x = x.reshape(batch_size, num_patches, self.patch_size)

        x = self.linear2(self.linear1(x))
        output_lengths = torch.ceil(lengths / self.patch_size)

        return x, output_lengths.to(dtype=lengths.dtype)


class WaveformUnpatcher(nn.Module):
    r"""Reconstructs waveform from patch embeddings.

    Args:
        patch_size (int): Number of audio samples per patch.
        input_dim (int): Input embedding dimension from the decoder.
        init_scale (float): Initial value for learnable output scale parameter.
    """

    def __init__(self, patch_size: int, input_dim: int, init_scale: float = 0.05):
        super().__init__()

        self.patch_size = patch_size
        self.input_dim = input_dim

        self.linear1 = nn.Linear(input_dim, input_dim, bias=True)
        self.linear2 = nn.Linear(input_dim, patch_size, bias=False)

        raw_init = math.log(math.exp(max(init_scale, 1e-6)) - 1)
        self.raw_output_scale = nn.Parameter(torch.tensor(raw_init))

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        max_output_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the waveform unpatcher.

        Args:
            features (Tensor): Input tensor with shape `(B, N, D)` where N is the
                number of patches and D is the embedding dimension.
            lengths (Tensor): Feature sequence lengths (number of valid patches)
                with shape `(B,)`.
            max_output_length (int, optional): Maximum output audio length for exact
                reconstruction. If provided, output will be trimmed/padded to this length.

        Returns:
            Tensor: Output waveform tensor with shape `(B, T, 1)` where T is either
                `max_output_length` or `N * patch_size`.
            Tensor: Output audio lengths with shape `(B,)`.
        """
        x = self.linear2(self.linear1(features))
        x = x * F.softplus(self.raw_output_scale)

        x = x.reshape(features.size(0), -1)
        output_lengths = lengths * self.patch_size

        if max_output_length is not None:
            x = F.pad(x, (0, max_output_length - x.size(1)))
            output_lengths = torch.clamp(output_lengths, max=max_output_length)

        return x.unsqueeze(2), output_lengths


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


class AudioChunkProcessor(nn.Module):
    """Audio chunk processor for streaming applications.

    This class enables chunk-based streaming audio processing using STFT/iSTFT.
    It manages overlap between chunks to ensure continuous audio processing.

    Args:
        window_size (int): Size of the FFT window.
        hop_length_ratio (int, optional): Ratio of window_size to hop_length.
            Default is 4, resulting in a hop length of window_size // hop_length_ratio.
        frames_per_chunk (int, optional): Number of frames to process in each chunk.
            Default is 16.
        power (Optional[float], optional): Power of the complex norm.
            If None, returns complex spectrogram. Default is None.
    """

    def __init__(
        self,
        window_size: int,
        hop_length_ratio: int = 4,
        frames_per_chunk: int = 16,
        power: Optional[float] = None,
    ):
        super().__init__()

        assert window_size > 0, "window_size must be positive"
        assert window_size % hop_length_ratio == 0, (
            "window_size must be divisible by hop_length_ratio."
        )
        assert hop_length_ratio > 1, "hop_length_ratio must be greater than 1."
        assert frames_per_chunk > 0, "frames_per_chunk must be positive"

        self.window_size = window_size
        self.hop_length_ratio = hop_length_ratio
        self.hop_length = window_size // hop_length_ratio
        self.frames_per_chunk = frames_per_chunk

        # Calculate chunk sizes
        self.overlap = window_size - self.hop_length  # Overlap between chunks
        self.valid_size = (
            self.hop_length * (frames_per_chunk - 1) + window_size - self.overlap
        )
        self.chunk_size = self.valid_size + self.overlap

        # Initialize spectrogram transforms
        self.specgram = TAudio.Spectrogram(
            n_fft=window_size,
            win_length=window_size,
            hop_length=self.hop_length,
            power=power,
            center=True,
        )

        self.inverse_specgram = TAudio.InverseSpectrogram(
            n_fft=window_size,
            win_length=window_size,
            hop_length=self.hop_length,
            center=True,
        )

    def forward(
        self, audio: torch.Tensor, return_full: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process a complete audio stream in chunks.

        Args:
            audio (Tensor): Input audio with shape (B, 1, T).
            return_full (bool, optional): Whether to return both chunk-based and
                full processing results for comparison. Default is False.

        Returns:
            torch.Tensor: Reconstructed audio from chunk-based processing.
            torch.Tensor (optional): Reconstructed audio from full processing if return_full is True.
        """
        audio_length = audio.size(-1)

        # Chunk-based streaming
        recon_chunks = []
        pointer = 0

        while pointer < audio_length:
            # Input chunk = valid_size + overlap
            chunk = audio[..., pointer : pointer + self.chunk_size]
            # Process chunk
            valid = self._process_chunk(chunk)
            recon_chunks.append(valid)
            # Move pointer
            pointer += self.valid_size

        # Stitch chunks back together
        recon_streaming = torch.cat(recon_chunks, dim=-1)[..., :audio_length]

        if return_full:
            # Full sequence processing for comparison
            spec_full = self.specgram(audio)
            recon_full = self.inverse_specgram(spec_full, length=audio_length)
            return recon_streaming, recon_full

        return recon_streaming

    def _process_chunk(
        self, chunk: torch.Tensor, return_spectrogram: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process a single audio chunk.

        Args:
            chunk (Tensor): Input audio chunk with shape (B, 1, T).
            return_spectrogram (bool, optional): Whether to return the spectrogram
                alongside the reconstructed audio. Default is False.

        Returns:
            torch.Tensor: Reconstructed audio chunk with valid samples.
            torch.Tensor (optional): Spectrogram of the chunk if return_spectrogram is True.
        """
        # Ensure chunk has correct size
        pad_len = self.chunk_size - chunk.size(-1)

        if pad_len > 0:
            chunk = F.pad(chunk, (0, pad_len))

        # STFT/iSTFT per chunk
        spec = self.specgram(chunk)
        recon = self.inverse_specgram(spec)

        # Get valid part (excluding lookahead at the end)
        valid = recon[..., : self.valid_size]

        if return_spectrogram:
            return valid, spec

        return valid
