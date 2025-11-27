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

from typing import List, Tuple

import torch
import torch.nn as nn

from soviamate.layers.conformer import ConformerLayer
from soviamate.layers.processor import InverseSpectrogramProcessor
from soviamate.utils.helper import (
    make_attention_mask,
    make_padding_mask,
    sample_chunk_config,
)


class AudioDecoder(nn.Module):
    r"""Audio Decoder with streaming inference capabilities.

    Uses dynamic chunk training for unified streaming and non-streaming models.

    Args:
        window_size (int): window size for the input frames.
        num_layers (int): number of conformer layers.
        d_model (int): embedding dimension for the conformer layers.
        ffn_dim (int): hidden dimension for the feed-forward module.
        num_heads (int): number of attention heads.
        kernel_size (int): kernel size for the convolutional module.
        dropout (float): dropout probability for each module.
        dynamic_chunk_sizes (List[int]): chunk sizes to sample from during training.
        left_context_ratio (int, optional): ratio of left_context to chunk_size. Default: 4.
        full_context_prob (float, optional): probability of full context mode. Default: 0.0.
        use_cross_attn (bool, optional): use cross-attention module. Default: False.
        cross_attn_dim (int, optional): dimension of cross-attention. Default: 256.
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
        dynamic_chunk_sizes: List[int],
        left_context_ratio: int = 4,
        full_context_prob: float = 0.0,
        use_cross_attn: bool = False,
        cross_attn_dim: int = 256,
    ):
        super().__init__()

        self.streaming_chunk_size = None
        self.streaming_left_context = None

        self.window_size = window_size
        self.embed_dim = d_model
        self.kernel_size = kernel_size
        self.use_cross_attn = use_cross_attn

        self.dynamic_chunk_sizes = dynamic_chunk_sizes
        self.left_context_ratio = left_context_ratio
        self.full_context_prob = full_context_prob

        self.layers = nn.ModuleList(
            [
                ConformerLayer(
                    d_model,
                    ffn_dim,
                    num_heads,
                    kernel_size,
                    dropout,
                    use_cross_attn,
                    cross_attn_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.inverse_specgram = InverseSpectrogramProcessor(
            window_size=window_size, input_dim=d_model, hop_length_ratio=4
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        embedding_lengths: torch.Tensor,
        prompts: torch.Tensor = None,
        prompt_lengths: torch.Tensor = None,
    ):
        r"""Forward pass of the audio decoder.

        During training, chunk sizes are randomly sampled. During inference,
        uses streaming config if set, otherwise uses full context.

        Args:
            embeddings (Tensor): input tensor with shape `(B, T, D)`.
            embedding_lengths (Tensor): length of the input tensor.
            prompts (Tensor, optional): prompt features with shape `(B, T_prompt, D_prompt)`.
            prompt_lengths (Tensor, optional): actual lengths of prompts with shape `(B,)`.

        Returns:
            Tensor: output tensor with shape `(B, T * chunk_size, 1)`.
            Tensor: length of the output tensor.
        """

        if embeddings.size(2) != self.embed_dim:
            raise ValueError("The embedding dimension does not match.")

        xs = embeddings
        x_lens = embedding_lengths

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

        attn_mask = make_attention_mask(x_lens, chunk_size, left_context)
        conv_mask = make_padding_mask(x_lens)

        zero_caches = self._initiate_states(xs.size(0), left_context, xs.device)

        prompt_mask = None
        if self.use_cross_attn and prompts is not None:
            prompt_mask = make_padding_mask(prompt_lengths)

        for layer, (conv_cache, attn_cache) in zip(self.layers, zero_caches):
            xs, _, _ = layer(
                xs, conv_mask, attn_mask, conv_cache, attn_cache, prompts, prompt_mask
            )

        xs, x_lens = self.inverse_specgram(xs, x_lens)

        return xs, x_lens

    @torch.jit.export
    def infer(
        self,
        segments: torch.Tensor,
        caches: List[List[torch.Tensor]] = None,
        prompts: torch.Tensor = None,
        prompt_lengths: torch.Tensor = None,
    ):
        r"""Inference for streaming audio input.

        Args:
            segments (Tensor): input tensor with shape `(B, chunk_size, D)`.
            caches (List[List[Tensor]]): list of lists of tensors representing
                internal state for each convolution and attention module.
            prompts (Tensor, optional): pre-computed prompt features with shape `(B, T', D')`.
                Should be computed once and reused for all chunks.
            prompt_lengths (Tensor, optional): actual lengths of prompts with shape `(B,)`.

        Returns:
            Tuple[Tensor, List[List[Tensor]]]:
                The output tensor and the updated caches.
        """

        device = segments.device
        batch_size = segments.size(0)

        if self.streaming_chunk_size is None or self.streaming_left_context is None:
            raise ValueError("The streaming configuration is not set.")

        if segments.size(1) != self.streaming_chunk_size:
            raise ValueError("The segment size does not match the chunk size.")

        if caches is None:
            caches = self._initiate_states(
                batch_size, self.streaming_left_context, device
            )

        xs = segments
        x_lens = torch.tensor([self.streaming_chunk_size] * batch_size, device=device)

        conv_mask = make_padding_mask(x_lens)
        attn_mask = make_attention_mask(
            x_lens, self.streaming_chunk_size, self.streaming_left_context
        )

        prompt_mask = None
        if self.use_cross_attn and prompts is not None:
            prompt_mask = make_padding_mask(prompt_lengths)

        new_caches = []
        for layer, (conv_cache, attn_cache) in zip(self.layers, caches):
            xs, conv_cache, attn_cache = layer(
                xs, conv_mask, attn_mask, conv_cache, attn_cache, prompts, prompt_mask
            )
            new_caches.append([conv_cache, attn_cache])

        xs, _ = self.inverse_specgram(xs, x_lens)

        return xs, new_caches

    @torch.jit.export
    def set_streaming_config(self, chunk_size: int):
        r"""Set the streaming configuration for the audio decoder.

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


class TextDecoder(nn.Module):
    r"""Text Decoder for speech recognition and other text processing tasks.

    Uses dynamic chunk training for unified streaming and non-streaming models.

    Args:
        output_dim (int): number of output classes.
        num_layers (int): number of conformer layers.
        d_model (int): embedding dimension for the conformer layers.
        ffn_dim (int): hidden dimension for the feed-forward module.
        num_heads (int): number of attention heads.
        kernel_size (int): kernel size for the convolutional module.
        dropout (float): dropout probability for the decoder.
        dynamic_chunk_sizes (List[int]): chunk sizes to sample from during training.
        left_context_ratio (int, optional): ratio of left_context to chunk_size. Default: 4.
        full_context_prob (float, optional): probability of full context mode. Default: 0.0.
    """

    def __init__(
        self,
        output_dim: int,
        num_layers: int,
        d_model: int,
        ffn_dim: int,
        num_heads: int,
        kernel_size: int,
        dropout: float,
        dynamic_chunk_sizes: List[int],
        left_context_ratio: int = 4,
        full_context_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.streaming_chunk_size = None
        self.streaming_left_context = None

        self.embed_dim = d_model
        self.kernel_size = kernel_size

        self.dynamic_chunk_sizes = dynamic_chunk_sizes
        self.left_context_ratio = left_context_ratio
        self.full_context_prob = full_context_prob

        self.layers = nn.ModuleList(
            [
                ConformerLayer(d_model, ffn_dim, num_heads, kernel_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.projector = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, output_dim)
        )

    def forward(
        self, encoder_outputs: torch.Tensor, encoder_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for the text decoder.

        During training, chunk sizes are randomly sampled. During inference,
        uses streaming config if set, otherwise uses full context.

        Args:
            encoder_outputs (Tensor): encoder outputs with shape `(B, T, D)`.
            encoder_lengths (Tensor): length of the input tensor with shape `(B,)`.

        Returns:
            features (Tensor): Features after Conformer blocks, before projection.
                Shape: (B, T, d_model). Used for RNN-T joint network.
            logits (Tensor): Projected outputs for CTC loss.
                Shape: (B, T, vocab_size).
            lengths (Tensor): Sequence lengths (B,).
        """

        if encoder_outputs.size(2) != self.embed_dim:
            raise ValueError("The embedding dimension does not match.")

        xs = encoder_outputs
        x_lens = encoder_lengths

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

        attn_mask = make_attention_mask(x_lens, chunk_size, left_context)
        conv_mask = make_padding_mask(x_lens)

        zero_caches = self._initiate_states(xs.size(0), left_context, xs.device)

        for layer, (conv_cache, attn_cache) in zip(self.layers, zero_caches):
            xs, _, _ = layer(xs, conv_mask, attn_mask, conv_cache, attn_cache)

        xs = self.projector(xs)

        return xs, x_lens

    @torch.jit.export
    def infer(self, segments: torch.Tensor, caches: None | List[List[torch.Tensor]]):
        r"""Inference for streaming text input.

        Args:
            segments (Tensor): input tensor with shape `(B, chunk_size, D)`.
            caches (List[List[Tensor]]): list of lists of tensors representing
                internal state for each convolution and attention module.

        Returns:
            Tuple[Tensor, List[List[Tensor]]]:
                The output tensor and the updated caches.
        """

        device = segments.device
        batch_size = segments.size(0)

        if self.streaming_chunk_size is None or self.streaming_left_context is None:
            raise ValueError("The streaming configuration is not set.")

        if segments.size(1) != self.streaming_chunk_size:
            raise ValueError("The segment size does not match the chunk size.")

        if caches is None:
            caches = self._initiate_states(
                batch_size, self.streaming_left_context, device
            )

        xs = segments
        x_lens = torch.tensor([self.streaming_chunk_size] * batch_size, device=device)

        conv_mask = make_padding_mask(x_lens)
        attn_mask = make_attention_mask(
            x_lens, self.streaming_chunk_size, self.streaming_left_context
        )

        new_caches = []
        for layer, cache in zip(self.layers, caches):
            xs, conv_cache, attn_cache = layer(
                xs, conv_mask, attn_mask, cache[0], cache[1]
            )
            new_caches.append([conv_cache, attn_cache])

        outputs = self.projector(xs)

        return outputs, new_caches

    @torch.jit.export
    def set_streaming_config(self, chunk_size: int):
        r"""Set the streaming configuration for the text decoder.

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
