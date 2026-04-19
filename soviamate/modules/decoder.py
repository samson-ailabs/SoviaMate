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

"""Decoder modules for reconstructing outputs from latent representations."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from soviamate.layers.processor import SpectralSynthesizer
from soviamate.layers.streaming import StreamingConformer


class AudioDecoder(StreamingConformer):
    r"""Audio decoder: streaming conformer layers followed by spectral synthesizer.

    Args:
        frame_stacking (int): Sub-frames per feature frame.
        hop_length (int): Samples per sub-frame (non-overlapping irfft window).
        num_layers (int): Number of conformer layers.
        d_model (int): Conformer embedding dimension.
        ffn_dim (int): Conformer feedforward hidden dimension.
        num_heads (int): Number of attention heads.
        kernel_size (int): Depthwise convolution kernel size.
        dropout (float): Dropout probability for each conformer sub-module.
        dynamic_chunk_sizes (List[int]): Chunk sizes sampled during training.
        left_context_ratio (int): Ratio of left_context to chunk_size. Default: ``4``.
        full_context_prob (float): Probability of full-context mode during training. Default: ``0.0``.
        speaker_dim (int): Speaker embedding dimension for AdaLN-Gaussian
            conditioning. Set to ``0`` to disable. Default: ``0``.
    """

    def __init__(
        self,
        frame_stacking: int,
        hop_length: int,
        num_layers: int,
        d_model: int,
        ffn_dim: int,
        num_heads: int,
        kernel_size: int,
        dropout: float,
        dynamic_chunk_sizes: List[int],
        left_context_ratio: int = 4,
        full_context_prob: float = 0.0,
        speaker_dim: int = 0,
    ):
        super().__init__(
            num_layers=num_layers,
            d_model=d_model,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dropout=dropout,
            dynamic_chunk_sizes=dynamic_chunk_sizes,
            left_context_ratio=left_context_ratio,
            full_context_prob=full_context_prob,
            speaker_dim=speaker_dim,
        )

        self.synthesizer = SpectralSynthesizer(
            frame_stacking=frame_stacking, hop_length=hop_length, input_dim=d_model
        )

    def forward(
        self,
        latents: torch.Tensor,
        lengths: torch.Tensor,
        speaker_emb: Optional[torch.Tensor] = None,
        max_output_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Full-context forward pass through the conformer layers and synthesizer.

        Args:
            latents (Tensor): Latent representations ``(B, T, D)``.
            lengths (Tensor): Per-sample valid frame counts ``(B,)``.
            speaker_emb (Tensor, optional): Speaker embedding for AdaLN ``(B, S)``.
            max_output_length (int, optional): Target output sample count.

        Returns:
            Tuple[Tensor, Tensor]: (waveforms, output_lengths) with shapes
                ``(B, T', 1)`` and ``(B,)``.
        """
        xs, x_lens = self._forward_layers(latents, lengths, speaker_emb)
        return self.synthesizer(xs, x_lens, max_output_length=max_output_length)

    @torch.jit.export
    def infer(
        self,
        segments: torch.Tensor,
        caches: Optional[List[List[torch.Tensor]]] = None,
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        r"""Streaming inference with caller-supplied state.

        Args:
            segments (Tensor): Latent chunk with shape
                ``(B, N * streaming_chunk_size, D)`` for any ``N >= 1``.
            caches (List[List[Tensor]]): Per-layer ``[conv_cache, attn_cache]``
                from the previous call, or ``None`` on a cold start.
            speaker_emb (Tensor, optional): Speaker embedding for AdaLN ``(B, S)``.

        Returns:
            Tuple[Tensor, List[List[Tensor]]]: raw waveform chunk and
                updated caches.
        """
        xs, new_caches = self._infer_layers(segments, caches, speaker_emb)
        lengths = torch.full(
            (segments.size(0),), xs.size(1), dtype=torch.int64, device=segments.device
        )
        waveforms, _ = self.synthesizer(xs, lengths)
        return waveforms, new_caches


class TextDecoder(StreamingConformer):
    r"""Text Decoder for CTC-based text supervision.

    Args:
        input_dim (int): input feature dimension (d_model from encoder).
        output_dim (int): number of output classes (vocab size).
        hidden_dim (int): hidden dimension for the conformer FFN and projector.
        num_layers (int): number of conformer refinement layers.
        num_heads (int): number of attention heads in conformer layers.
        kernel_size (int): kernel size for conformer convolution module.
        dropout (float): dropout probability.
        dynamic_chunk_sizes (List[int]): chunk sizes to sample from during training.
        left_context_ratio (int, optional): ratio of left_context to chunk_size. Default: 4.
        full_context_prob (float, optional): probability of full context mode. Default: 0.0.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        kernel_size: int,
        dropout: float,
        dynamic_chunk_sizes: List[int],
        left_context_ratio: int = 4,
        full_context_prob: float = 0.0,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            d_model=input_dim,
            ffn_dim=hidden_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dropout=dropout,
            dynamic_chunk_sizes=dynamic_chunk_sizes,
            left_context_ratio=left_context_ratio,
            full_context_prob=full_context_prob,
        )

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for the text decoder.

        Args:
            features (Tensor): input features with shape `(B, T, D)`.
            lengths (Tensor): valid lengths with shape `(B,)`.

        Returns:
            Tuple[Tensor, Tensor]: (logits, lengths) with shapes
                `(B, T, vocab_size)` and `(B,)`.
        """
        xs, x_lens = self._forward_layers(features, lengths)
        return self.projector(xs), x_lens

    @torch.jit.export
    def infer(
        self,
        segments: torch.Tensor,
        caches: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        r"""Streaming inference for the text decoder.

        Args:
            segments (Tensor): input tensor with shape `(B, chunk_size, D)`.
            caches (List[List[Tensor]]): convolution and attention caches per layer.

        Returns:
            Tuple[Tensor, List[List[Tensor]]]:
                Logits with shape `(B, chunk_size, vocab_size)` and updated caches.
        """
        xs, new_caches = self._infer_layers(segments, caches)
        return self.projector(xs), new_caches
