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
from torch.nn import functional as F
from torchaudio import transforms as T

from soviamate.layers.conformer import ConformerLayer
from soviamate.layers.redimnet import ASTP, Block1D, Block2D, Reshape, WeightSum
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

        self.linear = nn.Sequential(
            nn.Linear(window_size, d_model * 2, bias=False),
            nn.Linear(d_model * 2, d_model, bias=True),
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

        xs, x_lens = self._time_reduction(waveforms, lengths)
        xs = self.linear(xs)

        chunk_size, left_context = random.choice(self.attn_contexts)
        zero_caches = self._initiate_states(xs.size(0), left_context)

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

        batch_size = segments.size(0)
        valid_segment_length = self.chunk_size * self.window_size

        if self.chunk_size is None or self.left_context is None:
            raise ValueError("The streaming configuration is not set.")

        if segments.size(1) != valid_segment_length:
            raise ValueError("The segment size does not match the chunk size.")

        if segments.size(2) != 1:
            raise ValueError("The audio signal should be mono-channel.")

        if caches is None:
            caches = self._initiate_states(batch_size, self.left_context)

        lengths = torch.tensor(
            [valid_segment_length] * batch_size, device=segments.device
        )

        xs, x_lens = self._time_reduction(segments, lengths)
        xs = self.linear(xs)

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

    def _time_reduction(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Reduce the time dimension of the audio signal.

        Args:
            xs (Tensor): input tensor with shape `(batch, time, channel)`.
            x_lens (Tensor): length of the input tensor.

        Returns:
            Tuple[Tensor, Tensor]:
                The reduced input tensor and its length.
        """

        b, t, d = xs.shape
        n = t + (self.window_size - t % self.window_size) % self.window_size

        new_t = n // self.window_size
        new_d = d * self.window_size

        xs = F.pad(xs, (0, 0, 0, n - t))
        xs = xs.reshape(b, new_t, new_d).contiguous()

        x_lens = torch.div(x_lens - 1, self.window_size, rounding_mode="trunc")
        x_lens = (x_lens + 1).type(torch.long)

        return xs, x_lens


class SpeakerEncoder(nn.Module):
    r"""Speaker Encoder model for extracting speaker embeddings from speech signals.

    Args:
        sample_rate (int): sample rate of the input waveforms.
        n_fft (int): number of FFT points.
        win_length (int): window length in samples.
        hop_length (int): hop length in samples.
        num_channels (int): number of channels in the first layer.
        num_frequencies (int): number of mel frequencies.
        pooling_dim (int): dimension of the pooling layer.
        output_dim (int): dimension of the output embeddings.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        num_channels: int,
        num_frequencies: int,
        pooling_dim: int,
        output_dim: int,
    ) -> None:

        super().__init__()

        self.kernel_sizes_2d = [1, 3, 5, 7]
        self.kernel_sizes_1d = [7, 19, 31, 59]
        self.stage_setups = [
            [1, 4, 16],
            [2, 2, 16],
            [1, 2, 8],
            [4, 1, 8],
            [1, 1, 4],
            [8, 1, 4],
        ]

        self.hop_length = hop_length
        self.num_channels = num_channels
        self.num_frequencies = num_frequencies
        self.volumes = num_channels * num_frequencies

        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=num_frequencies,
        )

        self.stem_2d = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
        )

        for idx, (stride, factor_2d, factor_1d) in enumerate(self.stage_setups):
            num_channels = self.num_channels * stride
            num_frequencies = self.num_frequencies // stride

            layers = nn.ModuleList(
                [
                    Reshape(2, num_channels, num_frequencies),
                    Block2D(num_channels, self.kernel_sizes_2d, factor_2d),
                    Reshape(1, num_channels, num_frequencies),
                    Block1D(self.volumes, self.kernel_sizes_1d, factor_1d),
                    WeightSum(idx + 2, self.volumes),
                ]
            )

            setattr(self, f"block{idx}", layers)

        self.frm_pool = nn.Conv1d(self.volumes, output_dim, kernel_size=1)
        self.utt_pool = ASTP(self.volumes, pooling_dim, output_dim)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the model.

        Args:
            waveforms (Tensor): input waveforms with shape (B, 1, T).
            lengths (Tensor): lengths of the input waveforms with shape (B,).

        Returns:
            Tensor: frame-level embeddings with shape (B, T, D).
            Tensor: utterance-level embeddings with shape (B, 1, D).
        """

        xs = self.melspec(waveforms)
        x_lens = lengths // self.hop_length + 1

        xs = self.stem_2d(xs)
        xs = xs.flatten(1, 2)

        cache = [xs]
        for idx in range(len(self.stage_setups)):
            block = getattr(self, f"block{idx}")
            reshape_2d, block_2d, reshape_1d, block_1d, weight_sum = block

            xs = reshape_2d(xs)
            xs = block_2d(xs, x_lens)

            xs = reshape_1d(xs)
            xs = block_1d(xs, x_lens)

            cache.append(xs)
            xs = weight_sum(cache)

        mask = make_padding_mask(x_lens)
        mask = mask.unsqueeze(1)

        utt_embs = self.utt_pool(xs, x_lens).transpose(1, 2)
        frm_embs = self.frm_pool(xs).transpose(1, 2)

        return utt_embs, frm_embs, x_lens
