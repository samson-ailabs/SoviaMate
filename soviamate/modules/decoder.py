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

"""Multimodal Decoder for audio and visual inputs"""

import random
from typing import List, Tuple

import torch
import torch.nn as nn

from soviamate.layers.conformer import ConformerLayer
from soviamate.utils.helper import make_padding_mask, make_attention_mask


class AudioDecoder(nn.Module):
    r"""Audio Decoder with streaming inference capabilities.
    Suitable for speech synthesis, speech enhancement, and other speech processing tasks.

    Args:
        window_size (int): window size for the input frames.
        num_layers (int): number of conformer layers.
        d_model (int): embedding dimension for the conformer layers.
        ffn_dim (int): hidden dimension for the feed-forward module.
        num_heads (int): number of attention heads.
        kernel_size (int): kernel size for the convolutional module.
        dropout (float): dropout probability for each module.
        contexts (List[Tuple[int, int]]): list of tuples representing
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

        self.layers = nn.ModuleList(
            [
                ConformerLayer(d_model, ffn_dim, num_heads, kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=True),
            nn.Linear(2 * d_model, window_size, bias=False),
        )

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor):
        r"""Forward pass of the audio decoder.

        Args:
            embeddings (Tensor): input tensor with shape `(B, T, D)`.
            lengths (Tensor): length of the input tensor.

        Returns:
            Tensor: output tensor with shape `(B, T * chunk_size, 1)`.
            Tensor: length of the output tensor.
        """

        if embeddings.size(2) != self.embed_dim:
            raise ValueError("The embedding dimension does not match.")

        xs = embeddings
        x_lens = lengths

        chunk_size, left_context = random.choice(self.attn_contexts)
        zero_caches = self._initiate_states(xs.size(0), left_context)

        conv_mask = make_padding_mask(x_lens)
        attn_mask = make_attention_mask(x_lens, chunk_size, left_context)

        for layer, (conv_cache, attn_cache) in zip(self.layers, zero_caches):
            xs, _, _ = layer(xs, conv_mask, attn_mask, conv_cache, attn_cache)

        xs = self.linear(xs).flatten(1, 2).unsqueeze(-1)
        x_lens = (x_lens * self.window_size).type(torch.long)

        return xs, x_lens

    @torch.jit.export
    def infer(self, segments: torch.Tensor, caches: None | List[List[torch.Tensor]]):
        r"""Inference for streaming audio input.

        Args:
            segment (Tensor): input tensor with shape `(B, chunk_size, D)`.
            caches (List[List[Tensor]]): list of lists of tensors representing
                internal state for each convolution and attention module.

        Returns:
            Tuple[Tensor, List[List[Tensor]]]:
                The output tensor and the updated caches.
        """

        device = segments.device
        batch_size = segments.size(0)

        if self.chunk_size is None or self.left_context is None:
            raise ValueError("The streaming configuration is not set.")

        if segments.size(1) != self.chunk_size:
            raise ValueError("The segment size does not match the chunk size.")

        if caches is None:
            caches = self._initiate_states(batch_size, self.left_context)

        xs = segments
        x_lens = torch.tensor([self.chunk_size] * batch_size, device=device)

        conv_mask = make_padding_mask(x_lens)
        attn_mask = make_attention_mask(x_lens, self.chunk_size, self.left_context)

        new_caches = []
        for layer, cache in zip(self.layers, caches):
            xs, conv_cache, attn_cache = layer(
                xs, conv_mask, attn_mask, cache[0], cache[1]
            )
            new_caches.append([conv_cache, attn_cache])

        xs = self.linear(xs).flatten(1, 2).unsqueeze(-1)

        return xs, new_caches

    @torch.jit.export
    def set_streaming_config(self, chunk_size: int, left_context: int):
        r"""Set the streaming configuration for the audio decoder.

        Args:
            chunk_size (int): segment length for streaming inference.
            left_context (int): left context length for the attention module.
        """

        if [chunk_size, left_context] not in self.attn_contexts:
            raise ValueError("The size is not in the list of valid contexts.")

        self.chunk_size = chunk_size
        self.left_context = left_context

    def _initiate_states(
        self, batch_size: int, left_context: int
    ) -> List[List[torch.Tensor]]:
        r"""Initiate empty states for streaming inference.

        Args:
            batch_size (int): batch size for the input.
            left_context (int): left context length for the attention module.

        Returns:
            List[List[torch.Tensor]]: list of lists of tensors representing
                internal state for each convolution and attention module.
        """

        conv_cache = torch.zeros(batch_size, self.embed_dim, self.kernel_size - 1)
        attn_cache = torch.zeros(batch_size, left_context, self.embed_dim)

        return [[conv_cache, attn_cache] for _ in range(len(self.layers))]


class TextDecoder(nn.Module):
    r"""Text Decoder for speech recognition and other text processing tasks.

    Args:
        input_dim (int): output dimension of the encoder.
        hidden_dim (int): hidden dimension for the decoder.
        output_dim (int): number of output classes.
        dropout (float): dropout probability for the decoder.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, encoder_outpus: torch.Tensor) -> torch.Tensor:
        r"""Forward pass for the text decoder.

        Args:
            encoder_outpus (Tensor): encoder outputs with shape `(B, T, D)`.

        Returns:
            Tensor: output tensor with shape `(B, T, V)`.
        """

        return self.network(encoder_outpus)
