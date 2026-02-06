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

"""Vector Quantization for Discrete Representation Learning"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn


class FiniteScalarQuantizer(nn.Module):
    """Finite Scalar Quantizer.

    FSQ uses fixed-grid quantization without learnable codebooks,
    avoiding collapse issues common in VQ-VAE.

    Args:
        input_dim (int): Input feature dimension from encoder.
        fsq_levels (List[int]): FSQ quantization levels per dimension.
        noise_dropout (float): Dropout probability for FSQ noise approximation.
    """

    def __init__(self, input_dim: int, fsq_levels: List[int], noise_dropout: float):
        super().__init__()

        self.noise_dropout = noise_dropout
        self.codebook_size = math.prod(fsq_levels)
        self.bits_per_token = math.ceil(math.log2(self.codebook_size))

        _levels = torch.tensor(fsq_levels, dtype=torch.int64)
        self.register_buffer("levels", _levels, persistent=False)

        _basis = torch.cumprod(
            torch.tensor([1] + fsq_levels[:-1]), dim=0, dtype=torch.int64
        )
        self.register_buffer("basis", _basis, persistent=False)

        self.pre_quant = nn.Linear(input_dim, len(fsq_levels))
        self.post_quant = nn.Linear(len(fsq_levels), input_dim)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with FSQ quantization.

        Args:
            features (Tensor): features from encoder, shape `(B, T, D)`.
            lengths (Tensor): lengths of features, shape `(B,)`.

        Returns:
            Tensor: quantized features, shape `(B, T, D)`.
            Tensor: lengths of features, shape `(B,)`.
        """

        # Project directly to FSQ dimension
        z_fsq_input = self.pre_quant(features)

        # FSQ quantization
        z_quantized = self._quantize_vectors(z_fsq_input)

        # Output Projection
        quantized_features = self.post_quant(z_quantized)

        return quantized_features, lengths

    @torch.autocast(device_type="cuda", enabled=False)
    def _quantize_vectors(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x.type(torch.float))
        noise = (torch.rand_like(x) - 0.5) * 2 / (self.levels - 1)

        qx = self._inverse_scale_and_shift(
            self._straight_through_gradient(self._scale_and_shift(x))
        )

        if self.training and self.noise_dropout > 0:
            mask = self._generate_random_mask(x)
            qx = torch.where(mask, x, qx)

            mask = self._generate_random_mask(x)
            qx = torch.where(mask, qx, x + noise)

        return qx

    @torch.jit.export
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode input features to discrete indices.

        Args:
            inputs (Tensor): input tensor, shape `(B, T, D)`.

        Returns:
            Tensor: discrete indices, shape `(B, T, 1)`.
        """

        # Direct FSQ encoding
        z_fsq_input = self.pre_quant(inputs)

        # Convert to indices
        z_quantized = self._quantize_vectors(z_fsq_input)
        indices = self._codes_to_indices(z_quantized)

        return indices

    @torch.jit.export
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete indices to features.

        Args:
            indices (Tensor): discrete indices, shape `(B, T, 1)`.

        Returns:
            Tensor: reconstructed features, shape `(B, T, D)`.
        """

        # Convert indices to codes
        codes = self._indices_to_codes(indices)
        outputs = self.post_quant(codes)

        return outputs

    def _straight_through_gradient(self, z: torch.Tensor) -> torch.Tensor:
        return z + (z.round() - z).detach()

    def _generate_random_mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.full((x.size(0),), self.noise_dropout, device=x.device)
        mask = torch.bernoulli(mask)[:, None, None].bool().expand_as(x)
        return mask

    def _scale_and_shift(self, z: torch.Tensor) -> torch.Tensor:
        return (z + 1) * (self.levels - 1) / 2

    def _inverse_scale_and_shift(self, idxs: torch.Tensor) -> torch.Tensor:
        return idxs * 2 / (self.levels - 1) - 1

    def _indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        level_indices = (indices.unsqueeze(-1) // self.basis) % self.levels
        return self._inverse_scale_and_shift(level_indices)

    def _codes_to_indices(self, qx: torch.Tensor) -> torch.Tensor:
        level_indices = self._scale_and_shift(qx).round().to(torch.int64)
        return (level_indices * self.basis).sum(dim=-1)
