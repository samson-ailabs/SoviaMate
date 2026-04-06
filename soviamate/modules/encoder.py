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

from typing import List, Tuple

import torch

from soviamate.layers.processor import SpectrogramProcessor
from soviamate.layers.streaming import StreamingConformer


class AudioEncoder(StreamingConformer):
    """Audio Encoder with STFT feature extraction and conformer layers.

    Args:
        frame_stacking (int): number of spectrogram frames to stack.
        hop_length (int): hop length for STFT. Window length is ``2 * hop_length``.
        num_layers (int): number of conformer layers.
        d_model (int): embedding dimension for the conformer layers.
        ffn_dim (int): hidden dimension for the feed-forward module.
        num_heads (int): number of attention heads.
        kernel_size (int): kernel size for the convolutional module.
        dropout (float): dropout probability for each module.
        dynamic_chunk_sizes (List[int]): chunk sizes to sample from during training.
        left_context_ratio (int, optional): ratio of left_context to chunk_size. Default: 4.
        full_context_prob (float, optional): probability of full context mode. Default: 0.0.
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

        self.extractor = SpectrogramProcessor(
            frame_stacking=frame_stacking, hop_length=hop_length, output_dim=d_model
        )

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the audio encoder.

        During training, chunk sizes are randomly sampled. During inference,
        uses streaming config if set, otherwise uses full context.

        Args:
            waveforms (Tensor): input tensor with shape `(B, T, 1)`.
            lengths (Tensor): valid lengths in samples `(B,)`.

        Returns:
            Tuple[Tensor, Tensor]: (features, lengths) with shapes
                `(B, T', D)` and `(B,)`.
        """
        xs, x_lens = self.extractor(waveforms, lengths)
        return self._forward_layers(xs, x_lens)

    @torch.jit.export
    def infer(
        self, segments: torch.Tensor, caches: List[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        r"""Streaming inference for the audio encoder.

        Args:
            segments (Tensor): waveform chunk with shape
                ``(B, chunk_size * frame_stacking * hop_length, 1)``.
            caches (List[List[Tensor]]): convolution and attention caches per layer.

        Returns:
            Tuple[Tensor, List[List[Tensor]]]:
                Encoded features and updated caches.
        """
        lengths = torch.tensor(
            [segments.size(1)] * segments.size(0), device=segments.device
        )

        xs, _ = self.extractor(segments, lengths)
        xs = xs[:, : self.streaming_chunk_size, :]

        return self._infer_layers(xs, caches)
