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

"""Multimodal Encoder for audio and visual inputs"""

import random
from typing import List, Tuple

import torch
import torch.nn as nn

from soviamate.layers.conformer import ConformerLayer
from soviamate.layers.processor import SpectrogramProcessor
from soviamate.utils.helper import make_attention_mask, make_padding_mask


class AudioEncoder(nn.Module):
    """Audio Encoder with streaming inference capabilities.
    Suitable for speech recognition, speaker diarization, and other speech processing tasks.

    Args:
        window_size (int): size of the window for input frames.
        num_layers (int): number of conformer layers.
        d_model (int): embedding dimension for the conformer layers.
        ffn_dim (int): hidden dimension for the feed-forward module.
        num_heads (int): number of attention heads.
        kernel_size (int): kernel size for the convolutional module.
        dropout (float): dropout probability for each module.
        contexts (List[Tuple[int, int]]): List of tuples representing
            segment length and left context length for streaming inference.
    """

    def __init__(
        self,
        window_size: int,
        num_layers: int,
        d_model: int,
        ffn_dim: int,
        num_heads: int,
        kernel_size: int,
        dropout: int,
        contexts: List[Tuple[int, int]],
    ):
        super().__init__()

        self.chunk_size = None
        self.left_context = None

        for segment_length, left_context_length in contexts:
            if segment_length <= 0 or left_context_length <= 0:
                raise ValueError("The context size should be positive integers.")

            if left_context_length % segment_length != 0:
                raise ValueError(
                    "left_context_length should be divisible by segment_length"
                )

        self.window_size = window_size
        self.embed_dim = d_model
        self.kernel_size = kernel_size
        self.attn_contexts = contexts

        self.specgram = SpectrogramProcessor(
            window_size=window_size, output_dim=d_model, hop_length_ratio=4
        )

        self.layers = nn.ModuleList(
            [
                ConformerLayer(d_model, ffn_dim, num_heads, kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        r"""Forward pass of the audio encoder.

        Args:
            waveforms (Tensor): input tensor with shape `(B, T, 1)`.
            lengths (Tensor): length of the input tensor.

        Returns:
            Tensor: output tensor with shape `(B, T // window_size, D)`.
            Tensor: length of the output tensor.
        """

        if waveforms.size(2) != 1:
            raise ValueError("The audio signal should be mono-channel.")

        batch_size, device = waveforms.size(0), waveforms.device

        chunk_size, left_context = random.choice(self.attn_contexts)
        zero_caches = self._initiate_states(batch_size, left_context, device)

        xs, x_lens = self.specgram(waveforms, lengths)

        conv_mask = make_padding_mask(x_lens)
        attn_mask = make_attention_mask(x_lens, chunk_size, left_context)

        for layer, (conv_cache, attn_cache) in zip(self.layers, zero_caches):
            xs, _, _ = layer(xs, conv_mask, attn_mask, conv_cache, attn_cache)

        return xs, x_lens

    @torch.jit.export
    def infer(self, segments: torch.Tensor, caches: None | List[List[torch.Tensor]]):
        r"""Inference for streaming audio input.

        Args:
            segment (Tensor): input tensor with shape `(B, chunk_size, 1)`.
            caches (List[List[Tensor]]): list of lists of tensors representing
                internal state for each convolution and attention module.

        Returns:
            Tuple[Tensor, List[List[Tensor]]]:
                The output tensor and the updated caches.
        """

        batch_size, device = segments.size(0), segments.device
        valid_segment_length = self.chunk_size * self.window_size

        if self.chunk_size is None or self.left_context is None:
            raise ValueError("The streaming configuration is not set.")

        if segments.size(1) != valid_segment_length:
            raise ValueError("The segment size does not match the chunk size.")

        if segments.size(2) != 1:
            raise ValueError("The audio signal should be mono-channel.")

        if caches is None:
            caches = self._initiate_states(batch_size, self.left_context, device)

        lengths = torch.tensor(
            [valid_segment_length] * batch_size, device=segments.device
        )

        xs, x_lens = self.specgram(segments, lengths)

        conv_mask = make_padding_mask(x_lens)
        attn_mask = make_attention_mask(x_lens, self.chunk_size, self.left_context)

        new_caches = []
        for layer, cache in zip(self.layers, caches):
            xs, conv_cache, attn_cache = layer(
                xs, conv_mask, attn_mask, cache[0], cache[1]
            )
            new_caches.append([conv_cache, attn_cache])

        return xs, new_caches

    @torch.jit.export
    def set_streaming_config(self, chunk_size: int, left_context: int):
        r"""Set the streaming configuration for the audio encoder.

        Args:
            chunk_size (int): segment length for streaming inference.
            left_context (int): left context length for the attention module.
        """

        if [chunk_size, left_context] not in self.attn_contexts:
            raise ValueError("The size is not in the list of valid contexts.")

        self.chunk_size = chunk_size
        self.left_context = left_context

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

        conv_cache = torch.zeros(
            batch_size, self.embed_dim, self.kernel_size - 1, device=device
        )
        attn_cache = torch.zeros(
            batch_size, left_context, self.embed_dim, device=device
        )

        return [[conv_cache, attn_cache] for _ in range(len(self.layers))]
