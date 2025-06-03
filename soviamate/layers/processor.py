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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio.transforms as TAudio
from torch.nn import functional as F


class SpectrogramProcessor(nn.Module):
    r"""Spectrogram extraction processor with frame stacking capabilities.

    Args:
        window_size (int): Size of the window for input frames.
        hop_length_ratio (int, optional): Ratio of window_size to hop_length.
            Default is 4, resulting in a hop length of window_size // 4.
        output_dim (int, optional): Output dimension after linear projection.
            If None, no projection is applied. Default is None.
    """

    def __init__(
        self, window_size: int, hop_length_ratio: int = 4, output_dim: int = None
    ):
        super().__init__()

        assert (
            window_size % hop_length_ratio == 0
        ), "window_size must be divisible by hop_length_ratio."

        assert (
            hop_length_ratio > 1
        ), "hop_length_ratio must be greater than 1 for frame stacking."

        self.window_size = window_size
        self.hop_length_ratio = hop_length_ratio

        self.specgram = TAudio.Spectrogram(
            n_fft=window_size,
            win_length=window_size,
            hop_length=window_size // hop_length_ratio,
            power=None,
        )

        if output_dim is not None:
            self.linear = nn.Linear((window_size + 2) * hop_length_ratio, output_dim)
        else:
            self.linear = None

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the spectrogram processor.

        Args:
            waveforms (Tensor): Input tensor with shape `(B, T, 1)`.
            lengths (Tensor): Length of the input tensor.

        Returns:
            Tensor: Output tensor with shape `(B, T // window_size, D)`.
            Tensor: Length of the output tensor.
        """

        if waveforms.size(2) != 1:
            raise ValueError("The audio signal should be mono-channel.")

        device, dtype = lengths.device, lengths.dtype
        hop_length = self.window_size // self.hop_length_ratio

        # Extract spectrograms
        specs = self.specgram(waveforms.squeeze(2)).transpose(1, 2)
        mag, phase = specs.abs().clamp(1e-9).log(), specs.angle()

        # Calculate output lengths
        features = torch.cat((mag, phase), dim=2)
        lengths = torch.floor(lengths / hop_length) + 1

        # Pad to ensure we can create complete stacks
        pad_size = (
            self.hop_length_ratio - (features.size(1) % self.hop_length_ratio)
        ) % self.hop_length_ratio

        if pad_size > 0:
            features = F.pad(features, (0, 0, 0, pad_size))

        # Stack frames
        batch_size, time_dim, feat_dim = features.shape
        features = features.reshape(
            batch_size,
            time_dim // self.hop_length_ratio,
            feat_dim * self.hop_length_ratio,
        )

        # Update lengths
        lengths = torch.ceil(lengths / self.hop_length_ratio)
        lengths = lengths.to(device=device, dtype=dtype)

        # Apply linear projection if specified
        if self.linear is not None:
            features = self.linear(features)

        return features, lengths


class InverseSpectrogramProcessor(nn.Module):
    r"""Inverse spectrogram processor for reconstructing waveforms from spectrograms.

    Args:
        window_size (int): Size of the window for output frames.
        hop_length_ratio (int, optional): Ratio of window_size to hop_length.
            Default is 4, resulting in a hop length of window_size // 4.
        input_dim (int, optional): Input dimension before linear projection.
            If None, no projection is applied. Default is None.
    """

    def __init__(
        self,
        window_size: int,
        hop_length_ratio: int = 4,
        input_dim: int = None,
    ):
        super().__init__()

        assert (
            window_size % hop_length_ratio == 0
        ), "window_size must be divisible by hop_length_ratio."

        assert (
            hop_length_ratio > 1
        ), "hop_length_ratio must be greater than 1 for frame stacking."

        self.window_size = window_size
        self.hop_length_ratio = hop_length_ratio

        if input_dim is not None:
            self.linear = nn.Linear(input_dim, (window_size + 2) * hop_length_ratio)
        else:
            self.linear = None

        self.inverse_specgram = TAudio.InverseSpectrogram(
            n_fft=window_size,
            win_length=window_size,
            hop_length=window_size // hop_length_ratio,
        )

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the inverse spectrogram processor.

        Args:
            features (Tensor): Input tensor with shape `(B, T, D)`.
            lengths (Tensor): Length of the input tensor.

        Returns:
            Tensor: Output waveform tensor with shape `(B, T * window_size, 1)`.
            Tensor: Length of the output tensor.
        """

        # Apply linear projection if specified
        if self.linear is not None:
            features = self.linear(features)

        # Unstack frames
        batch_size, time_dim, feat_dim = features.shape
        features = features.reshape(
            batch_size,
            time_dim * self.hop_length_ratio,
            feat_dim // self.hop_length_ratio,
        )

        # Update lengths
        lengths = lengths * self.hop_length_ratio

        # Split into magnitude and phase
        mag, phase = features.split(self.window_size // 2 + 1, dim=2)

        # Convert to complex spectrogram
        real = mag.exp() * torch.cos(phase)
        imag = mag.exp() * torch.sin(phase)

        # Stack real and imaginary parts
        specs = torch.stack([real, imag], dim=-1)
        specs = torch.view_as_complex(specs.contiguous())

        # Convert to waveform
        waveforms = self.inverse_specgram(specs.transpose(1, 2)).unsqueeze(2)

        # Calculate output lengths
        lengths = (lengths - 1) * (self.window_size // self.hop_length_ratio)

        return waveforms, lengths


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
        assert (
            window_size % hop_length_ratio == 0
        ), "window_size must be divisible by hop_length_ratio."
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
            audio (torch.Tensor): Input audio with shape (B, 1, T).
            return_full (bool, optional): Whether to return both chunk-based and
                full processing results for comparison. Default is False.

        Returns:
            torch.Tensor: Reconstructed audio from chunk-based processing.
            torch.Tensor (optional): Reconstructed audio from full processing if return_full is True.
        """
        batch_size = audio.size(0)
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
            chunk (torch.Tensor): Input audio chunk with shape (B, 1, T).
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
