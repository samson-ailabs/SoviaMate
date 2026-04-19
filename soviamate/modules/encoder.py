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

"""Encoder modules for transforming inputs to latent representations."""

from typing import List, Optional, Tuple

import torch

from soviamate.layers.processor import SpectralAnalyzer
from soviamate.layers.streaming import StreamingConformer


class AudioEncoder(StreamingConformer):
    r"""Audio encoder: spectral analyzer followed by streaming conformer layers.

    Args:
        frame_stacking (int): Sub-frames per feature frame.
        hop_length (int): Samples per sub-frame (non-overlapping rfft window).
        num_layers (int): Number of conformer layers.
        d_model (int): Conformer embedding dimension.
        ffn_dim (int): Conformer feedforward hidden dimension.
        num_heads (int): Number of attention heads.
        kernel_size (int): Depthwise convolution kernel size.
        dropout (float): Dropout probability for each conformer sub-module.
        dynamic_chunk_sizes (List[int]): Chunk sizes sampled during training.
        left_context_ratio (int): Ratio of left_context to chunk_size. Default: ``4``.
        full_context_prob (float): Probability of full-context mode during training. Default: ``0.0``.
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
        )

        self.analyzer = SpectralAnalyzer(
            frame_stacking=frame_stacking, hop_length=hop_length, output_dim=d_model
        )

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Full-context forward pass through the analyzer and conformer layers.

        Args:
            waveforms (Tensor): Raw waveform ``(B, T, 1)``.
            lengths (Tensor): Per-sample valid sample counts ``(B,)``.

        Returns:
            Tuple[Tensor, Tensor]: (features, lengths) with shapes ``(B, T', D)`` and ``(B,)``.
        """
        xs, x_lens = self.analyzer(waveforms, lengths)
        return self._forward_layers(xs, x_lens)

    @torch.jit.export
    def infer(
        self,
        segments: torch.Tensor,
        caches: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        r"""Streaming inference with caller-supplied state.

        Args:
            segments (Tensor): Waveform
                ``(B, N * streaming_chunk_size * frame_stacking * hop_length, 1)``.
            caches (List[List[Tensor]]): Per-layer ``[conv_cache, attn_cache]``
                from the previous call, or ``None`` on a cold start.

        Returns:
            Tuple[Tensor, List[List[Tensor]]]: encoded features with shape
                ``(B, N * streaming_chunk_size, D)`` and updated caches.
        """
        batch_size, seq_len, _ = segments.size()
        lengths = torch.full(
            (batch_size,), seq_len, dtype=torch.int64, device=segments.device
        )
        xs, _ = self.analyzer(segments, lengths)
        return self._infer_layers(xs, caches)
