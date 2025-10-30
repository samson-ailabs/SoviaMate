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

"""Vector and Scalar Quantization"""

from typing import List, Tuple

import torch
import torch.nn as nn


class FiniteScalarQuantization(nn.Module):
    r"""Finite Scalar Quantization with symmetry-preserving and noise-approximated quantization
    based on *Scaling Transformers for Low-Bitrate High-Quality Speech Coding* by Stability AI.

    Args:
        input_dim (int): input dimension.
        levels (List[int]): set of quantization level numbers.
        num_codebooks (int): number of codebooks.
        noise_dropout (float): dropout probability for noise approximation.
    """

    def __init__(
        self,
        input_dim: int,
        levels: List[int],
        num_codebooks: int,
        noise_dropout: float,
    ):
        super().__init__()

        assert num_codebooks == 1, "Only support single codebook for now."

        self.num_codebooks = num_codebooks
        self.noise_dropout = noise_dropout

        _levels = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("levels", _levels, persistent=False)

        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int64
        )
        self.register_buffer("basis", _basis, persistent=False)

        self.proj_input = nn.Linear(input_dim, num_codebooks * len(levels))
        self.proj_output = nn.Linear(num_codebooks * len(levels), input_dim)

    @torch.autocast(device_type="cuda", enabled=False)
    def _quantize_vectors(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x.type(torch.float))
        noise = (torch.rand_like(x) - 0.5) * 2 / (self.levels - 1)

        qx = self._inverse_scale_and_shift(
            self._straight_through_gradient(self._scale_and_shift(x))
        )

        if self.training:
            mask = self._generate_random_mask(x)
            qx = torch.where(mask, x, qx)

            mask = self._generate_random_mask(x)
            qx = torch.where(mask, qx, x + noise)

        return qx

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the quantizer.

        Args:
            inputs (Tensor): input tensor, shape (B, T, D).
            lengths (Tensor): lengths of the input tensor, shape (B,).

        Returns:
            Tensor: quantized tensor, shape (B, T, D).
        """

        batch_size, seq_len, _ = inputs.size()

        latents = self.proj_input(inputs)
        latents = latents.reshape(batch_size, seq_len, self.num_codebooks, -1)

        codes = self._quantize_vectors(latents)
        codes = codes.reshape(batch_size, seq_len, -1)

        outputs = self.proj_output(codes)

        return outputs, lengths

    @torch.jit.export
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Encode input tensor to quantized tensor.

        Args:
            inputs (Tensor): input tensor, shape (B, T, D).

        Returns:
            Tensor: discrete indices, shape (B, T, 1).
        """

        batch_size, seq_len, _ = inputs.size()

        latents = self.proj_input(inputs)
        latents = latents.reshape(batch_size, seq_len, self.num_codebooks, -1)

        codes = self._quantize_vectors(latents)
        indices = self._codes_to_indices(codes)

        return indices

    @torch.jit.export
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        r"""Decode quantized tensor to original tensor.

        Args:
            indices (Tensor): discrete indices, shape (B, T, 1).

        Returns:
            Tensor: quantized tensor, shape (B, T, D).
        """

        batch_size, seq_len, num_codebooks = indices.size()
        codebook_dim = num_codebooks * len(self.levels)

        assert num_codebooks == self.num_codebooks, "Number of codebooks mismatched."

        codes = self._indices_to_codes(indices)
        codes = codes.reshape(batch_size, seq_len, codebook_dim)

        outputs = self.proj_output(codes)

        return outputs

    def _bound(self, z: torch.Tensor):
        half_l = (self.levels - 1) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        half_width = self.levels // 2
        return self._straight_through_gradient(bounded_z) / half_width

    def _straight_through_gradient(self, z: torch.Tensor):
        return z + (z.round() - z).detach()

    def _generate_random_mask(self, x: torch.Tensor):
        mask = torch.full((x.size(0),), self.noise_dropout, device=x.device)
        mask = torch.bernoulli(mask)[:, None, None, None].bool().expand_as(x)
        return mask

    def _scale_and_shift(self, z: torch.Tensor):
        return (z + 1) * (self.levels - 1) / 2

    def _inverse_scale_and_shift(self, idxs: torch.Tensor):
        return idxs * 2 / (self.levels - 1) - 1

    def _indices_to_level_indices(self, indices: torch.Tensor):
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self.basis) % self.levels
        return codes_non_centered

    def _indices_to_codes(self, indices: torch.Tensor):
        level_indices = self._indices_to_level_indices(indices)
        codes = self._inverse_scale_and_shift(level_indices)
        return codes

    def _codes_to_indices(self, qx: torch.Tensor):
        qx = self._scale_and_shift(qx)
        qx = (qx * self.basis).sum(dim=-1)
        idx = qx.round().type(torch.int64)
        return idx
