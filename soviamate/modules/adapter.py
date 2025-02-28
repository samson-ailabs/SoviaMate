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

"""Extracting embeddings using Adapter models"""

import torch
import torch.nn as nn
import torchaudio.transforms as T

from soviamate.layers.redimnet import ASTP, Block1D, Block2D, Reshape, WeightSum
from soviamate.utils.helper import make_padding_mask


class SpeakerAdapter(nn.Module):
    r"""Speaker Adapter model for extracting speaker embeddings from speech signals.

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
            Tensor: frame-level embeddings with shape (B, D, T).
            Tensor: utterance-level embeddings with shape (B, D, 1).
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

        utt_embs = self.utt_pool(xs, x_lens)
        frm_embs = self.frm_pool(xs).masked_fill(mask, 0)

        return utt_embs, frm_embs, x_lens
