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
import torch.nn as nn

from soviamate.layers.conformer import ConformerLayer
from soviamate.layers.processor import SpectrogramProcessor
from soviamate.utils.helper import (
    make_attention_mask,
    make_padding_mask,
    sample_chunk_config,
)


class AudioEncoder(nn.Module):
    """Audio Encoder with streaming inference capabilities.

    Uses dynamic chunk training for unified streaming and non-streaming models.
    Uses STFT-based spectrogram extraction with frame stacking.

    Args:
        frame_stacking (int): number of spectrogram frames to stack.
        window_length (int): window length for STFT (n_fft).
        hop_length (int): hop length for STFT.
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
        window_length: int,
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
        super().__init__()

        self.streaming_chunk_size = None
        self.streaming_left_context = None

        self.frame_stacking = frame_stacking
        self.window_length = window_length
        self.hop_length = hop_length

        self.embed_dim = d_model
        self.kernel_size = kernel_size

        self.dynamic_chunk_sizes = dynamic_chunk_sizes
        self.left_context_ratio = left_context_ratio
        self.full_context_prob = full_context_prob

        self.extractor = SpectrogramProcessor(
            frame_stacking=frame_stacking,
            window_length=window_length,
            hop_length=hop_length,
            output_dim=d_model,
        )

        self.layers = nn.ModuleList(
            [
                ConformerLayer(d_model, ffn_dim, num_heads, kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the audio encoder.

        During training, chunk sizes are randomly sampled. During inference,
        uses streaming config if set, otherwise uses full context.

        Args:
            waveforms (Tensor): input tensor with shape `(B, T, 1)`.
            waveform_lengths (Tensor): length of the input tensor.

        Returns:
            Tensor: output tensor with shape `(B, T // window_size, D)`.
            Tensor: length of the output tensor.
        """

        if waveforms.size(2) != 1:
            raise ValueError("The audio signal should be mono-channel.")

        xs, x_lens = self.extractor(waveforms, waveform_lengths)

        if self.training:
            chunk_size, left_context = sample_chunk_config(
                x_lens,
                self.dynamic_chunk_sizes,
                self.left_context_ratio,
                self.full_context_prob,
            )
        elif self.streaming_chunk_size is not None:
            chunk_size = self.streaming_chunk_size
            left_context = self.streaming_left_context
        else:
            chunk_size, left_context = xs.size(1), 0

        conv_mask = make_padding_mask(x_lens)
        attn_mask = make_attention_mask(x_lens, chunk_size, left_context)

        zero_caches = self._initiate_states(xs.size(0), left_context, xs.device)

        for layer, (conv_cache, attn_cache) in zip(self.layers, zero_caches):
            xs, _, _ = layer(xs, conv_mask, attn_mask, conv_cache, attn_cache)

        return xs, x_lens

    @torch.jit.export
    def infer(
        self, segments: torch.Tensor, caches: List[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        r"""Inference for streaming audio input.

        Args:
            segments (Tensor): input tensor with shape `(B, chunk_size, 1)`.
            caches (List[List[Tensor]]): list of lists of tensors representing
                internal state for each convolution and attention module.

        Returns:
            Tuple[Tensor, List[List[Tensor]]]:
                The output tensor and the updated caches.
        """

        batch_size, device = segments.size(0), segments.device
        valid_segment_length = (
            self.streaming_chunk_size * self.hop_length * self.frame_stacking
        )

        if self.streaming_chunk_size is None or self.streaming_left_context is None:
            raise ValueError("The streaming configuration is not set.")

        if segments.size(1) != valid_segment_length:
            raise ValueError(
                f"Segment size {segments.size(1)} does not match expected {valid_segment_length}."
            )

        if segments.size(2) != 1:
            raise ValueError("The audio signal should be mono-channel.")

        if caches is None:
            caches = self._initiate_states(
                batch_size, self.streaming_left_context, device
            )

        lengths = torch.tensor(
            [valid_segment_length] * batch_size, device=segments.device
        )

        xs, x_lens = self.extractor(segments, lengths)

        conv_mask = make_padding_mask(x_lens)
        attn_mask = make_attention_mask(
            x_lens, self.streaming_chunk_size, self.streaming_left_context
        )

        new_caches = []
        for layer, (conv_cache, attn_cache) in zip(self.layers, caches):
            xs, conv_cache, attn_cache = layer(
                xs, conv_mask, attn_mask, conv_cache, attn_cache
            )
            new_caches.append([conv_cache, attn_cache])

        return xs, new_caches

    @torch.jit.export
    def set_streaming_config(self, chunk_size: int):
        r"""Set the streaming configuration for the audio encoder.

        Args:
            chunk_size (int): segment length for streaming inference.
        """

        self.streaming_chunk_size = chunk_size
        self.streaming_left_context = chunk_size * self.left_context_ratio

    def _initiate_states(
        self, batch_size: int, left_context: int, device: torch.device
    ) -> List[List[torch.Tensor]]:
        r"""Initiate empty states for streaming inference.

        Args:
            batch_size (int): batch size for the input.
            left_context (int): left context length for the attention module.
            device (torch.device): device for the tensors.

        Returns:
            List[List[torch.Tensor]]: list of lists of tensors representing
                internal state for each convolution and attention module.
        """

        conv_left_context = self.kernel_size - 1 if left_context > 0 else 0

        conv_cache = torch.zeros(
            batch_size, self.embed_dim, conv_left_context, device=device
        )
        attn_cache = torch.zeros(
            batch_size, left_context, self.embed_dim, device=device
        )

        return [[conv_cache, attn_cache] for _ in range(len(self.layers))]
