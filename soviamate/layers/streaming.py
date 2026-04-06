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

"""Base class for streaming conformer modules."""

import abc
from typing import List, Tuple

import torch
import torch.nn as nn

from soviamate.layers.conformer import ConformerLayer
from soviamate.utils.helper import (
    make_attention_mask,
    make_padding_mask,
    sample_chunk_config,
)


class StreamingConformer(nn.Module, abc.ABC):
    """Base class providing shared streaming conformer infrastructure.

    Handles dynamic chunk training, streaming cache management, and the
    common forward/infer loop over conformer layers.

    Args:
        num_layers (int): number of conformer layers.
        d_model (int): embedding dimension.
        ffn_dim (int): hidden dimension for the feed-forward module.
        num_heads (int): number of attention heads.
        kernel_size (int): kernel size for the convolutional module.
        dropout (float): dropout probability.
        dynamic_chunk_sizes (List[int]): chunk sizes to sample from during training.
        left_context_ratio (int): ratio of left_context to chunk_size. Default: 4.
        full_context_prob (float): probability of full context mode. Default: 0.0.
        speaker_dim (int): speaker embedding dimension for AdaLN. Default: 0.
    """

    def __init__(
        self,
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
        super().__init__()

        self.streaming_chunk_size = None
        self.streaming_left_context = None

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dynamic_chunk_sizes = dynamic_chunk_sizes
        self.left_context_ratio = left_context_ratio
        self.full_context_prob = full_context_prob

        self.layers = nn.ModuleList(
            [
                ConformerLayer(
                    d_model, ffn_dim, num_heads, kernel_size, dropout, speaker_dim
                )
                for _ in range(num_layers)
            ]
        )

    @torch.jit.export
    def set_streaming_config(self, chunk_size: int):
        r"""Set the streaming configuration.

        Args:
            chunk_size (int): segment length for streaming inference.
        """
        self.streaming_chunk_size = chunk_size
        self.streaming_left_context = chunk_size * self.left_context_ratio

    def _forward_layers(
        self, xs: torch.Tensor, x_lens: torch.Tensor, spk_emb: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run conformer layers with dynamic chunk masking.

        Args:
            xs (Tensor): input features with shape `(B, T, D)`.
            x_lens (Tensor): valid lengths with shape `(B,)`.
            spk_emb (Tensor, optional): speaker embedding for AdaLN `(B, S)`.

        Returns:
            Tuple[Tensor, Tensor]: (output, lengths) with shapes `(B, T, D)` and `(B,)`.
        """
        if self.training and self.dynamic_chunk_sizes:
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

        caches = self._init_caches(xs.size(0), left_context, xs.device)

        for layer, (conv_cache, attn_cache) in zip(self.layers, caches):
            xs, _, _ = layer(xs, conv_mask, attn_mask, conv_cache, attn_cache, spk_emb)

        return xs, x_lens

    def _infer_layers(
        self,
        segments: torch.Tensor,
        caches: List[List[torch.Tensor]] = None,
        spk_emb: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """Run conformer layers in streaming mode with caches.

        Args:
            segments (Tensor): input chunk with shape `(B, chunk_size, D)`.
            caches (List[List[Tensor]], optional): per-layer [conv_cache, attn_cache].
            spk_emb (Tensor, optional): speaker embedding for AdaLN `(B, S)`.

        Returns:
            Tuple[Tensor, List[List[Tensor]]]: (output, new_caches).
        """
        batch_size = segments.size(0)
        device = segments.device

        if self.streaming_chunk_size is None or self.streaming_left_context is None:
            raise ValueError("Streaming configuration is not set.")

        if segments.size(1) != self.streaming_chunk_size:
            raise ValueError(
                f"Segment size {segments.size(1)} does not match "
                f"streaming_chunk_size {self.streaming_chunk_size}."
            )

        if caches is None:
            caches = self._init_caches(batch_size, self.streaming_left_context, device)

        xs = segments
        x_lens = torch.tensor([self.streaming_chunk_size] * batch_size, device=device)

        conv_mask = make_padding_mask(x_lens)
        attn_mask = make_attention_mask(
            x_lens, self.streaming_chunk_size, self.streaming_left_context
        )

        new_caches = []
        for layer, (conv_cache, attn_cache) in zip(self.layers, caches):
            xs, conv_cache, attn_cache = layer(
                xs, conv_mask, attn_mask, conv_cache, attn_cache, spk_emb
            )
            new_caches.append([conv_cache, attn_cache])

        return xs, new_caches

    def _init_caches(
        self, batch_size: int, left_context: int, device: torch.device
    ) -> List[List[torch.Tensor]]:
        r"""Create zero-initialized caches for streaming inference.

        Args:
            batch_size (int): batch size for the cache tensors.
            left_context (int): left context length for attention.
            device (torch.device): device for the tensors.

        Returns:
            List[List[Tensor]]: per-layer [conv_cache, attn_cache].
        """
        conv_left = self.kernel_size - 1 if left_context > 0 else 0

        conv_cache = torch.zeros(batch_size, self.d_model, conv_left, device=device)
        attn_cache = torch.zeros(batch_size, left_context, self.d_model, device=device)

        return [[conv_cache, attn_cache] for _ in range(len(self.layers))]
