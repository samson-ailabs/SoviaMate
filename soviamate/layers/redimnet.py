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

"""Reshape Dimensions Network for Speaker Recognition"""

from typing import List

import torch
import torch.nn as nn

from soviamate.utils.helper import make_padding_mask


class _ConvNeXtBlock(nn.Module):
    r"""ConvNeXt-like block for the 1D case.

    Args:
        num_channels (int): number of input channels
        kernel_size (int): kernel size of the convolutional layer
    """

    def __init__(self, num_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(
                num_channels,
                num_channels,
                kernel_size=kernel_size,
                padding="same",
                groups=num_channels,
            ),
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=1),
        )

    def forward(self, xs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the ConvNeXtBlock

        Args:
            xs (Tensor): input tensor, shape (B, C, T)
            mask (Tensor): padding mask, shape (B, C, T)

        Returns:
            Tensor: output tensor, shape (B, C, T)
        """

        xs = xs.masked_fill(mask, 0.0)
        xs = xs + self.network(xs)

        return xs


class Block1D(nn.Module):
    r"""1D sub-block based on ConvNeXt-like block and Transformer Encoder.

    Args:
        input_channels (int): number of input channels
        kernel_sizes (List[int]): list of kernel sizes for each layer
        factor (int): ratio to reduce the number of channels
    """

    def __init__(
        self, input_channels: int, kernel_sizes: List[int], factor: int
    ) -> None:

        super().__init__()

        hidden_channels = input_channels // factor

        self.red_dim_inp = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.exp_dim_out = nn.Conv1d(hidden_channels, input_channels, kernel_size=1)

        self.time_context = nn.ModuleList(
            [
                _ConvNeXtBlock(hidden_channels, kernel_size)
                for kernel_size in kernel_sizes
            ]
        )

        self.attention = nn.TransformerEncoderLayer(
            hidden_channels, 4, hidden_channels * 4, batch_first=True
        )

    def forward(self, xs: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the Block1D

        Args:
            xs (Tensor): input tensor, shape (B, C, T)
            x_lens (Tensor): length of the input sequences, shape (B,)

        Returns:
            Tensor: output tensor, shape (B, C, T)
        """

        residual = xs

        mask = make_padding_mask(x_lens)
        mask = mask.unsqueeze(1)

        xs = xs.masked_fill(mask, 0.0)
        xs = self.red_dim_inp(xs)

        for time_ctx in self.time_context:
            xs = time_ctx(xs, mask)

        xs = xs.permute(0, 2, 1)
        xs = self.attention(xs, None, mask.squeeze(1))
        xs = xs.permute(0, 2, 1)

        xs = xs.masked_fill(mask, 0.0)
        xs = self.exp_dim_out(xs)

        xs = residual + xs

        return xs


class Block2D(nn.Module):
    r"""2D sub-block based on Basic ResNet block.

    Args:
        input_channels (int): number of input channels
        kernel_sizes (List[int]): list of kernel sizes for each layer
        factor (int): ratio to expand the number of channels
    """

    def __init__(
        self, input_channels: int, kernel_sizes: List[int], factor: int
    ) -> None:

        super().__init__()

        hidden_channels = input_channels * factor

        self.exp_dim_inp = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.red_dim_out = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding="same",
                        groups=hidden_channels,
                    ),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding="same",
                        groups=hidden_channels,
                    ),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                    nn.BatchNorm2d(hidden_channels),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, xs: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the ResNetBlock

        Args:
            xs (Tensor): input tensor, shape (B, C, F, T)
            x_lens (Tensor): length of the input sequences, shape (B,)

        Returns:
            Tensor: output tensor, shape (B, C, F, T)
        """

        mask = make_padding_mask(x_lens)
        mask = mask.unsqueeze(1).unsqueeze(2)

        xs = xs.masked_fill(mask, 0.0)
        xs = self.exp_dim_inp(xs)

        for layer in self.layers:
            xs = xs.masked_fill(mask, 0.0)
            xs = xs + layer(xs)

        xs = xs.masked_fill(mask, 0.0)
        xs = self.red_dim_out(xs)

        return xs


class Reshape(nn.Module):
    r"""Reshape layer for the input tensor.

    Args:
        ndim (int): number of dimensions of the output tensor
        num_channels (int): number of channels of the input tensor
        num_freq_bins (int): number of frequency bins of the input tensor
    """

    def __init__(self, ndim: int, num_channels: int, num_freq_bins: int) -> None:
        super().__init__()

        self.ndim = ndim
        self.num_channels = num_channels
        self.num_freq_bins = num_freq_bins
        self.volume = num_channels * num_freq_bins

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the Reshape layer.

        Args:
            xs (Tensor): input tensor, shape (B, C, F, T) or (B, C * F, T)

        Returns:
            Tensor: output tensor, shape (B, C * F, T) or (B, C, F, T)
        """

        if self.ndim == 1:
            b, c, f, t = xs.shape
            assert (
                c == self.num_channels
            ), f"Invalid shape: expected {self.num_channels} channels, but got {c}"
            assert (
                f == self.num_freq_bins
            ), f"Invalid shape: expected {self.num_freq_bins} frequency bins, but got {f}"
            xs = xs.reshape(b, c * f, t)

        if self.ndim == 2:
            b, cf, t = xs.shape
            assert (
                cf == self.volume
            ), f"Invalid shape: expected {self.volume} volume, but got {cf}"
            xs = xs.reshape(b, self.num_channels, self.num_freq_bins, t)

        return xs


class WeightSum(nn.Module):
    r"""Weighted sum of 1D input features.

    Args:
        num_features (int): number of input features to combine
        num_channels (int): number of channels of the input tensor
    """

    def __init__(self, num_features: int, num_channels: int) -> None:
        super().__init__()

        self.weights = nn.Parameter(torch.ones(num_features, num_channels))

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        r"""Forward pass of the WeightSum1D layer.

        Args:
            xs (List[Tensor]): list of input tensors, shape (B, C, T)

        Returns:
            Tensor: output tensor, shape (B, C, T)
        """

        w = torch.softmax(self.weights, dim=0)

        xs = torch.stack(xs, dim=1)
        xs = torch.einsum("bnct,nc->bct", xs, w)

        return xs


class ASTP(nn.Module):
    """Attentive Statistics Pooling layer.

    Args:
        input_dim (int): number of input channels
        hidden_dim (int): number of hidden channels
        output_dim (int): number of output channels
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        self.inp_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
        )

        self.network = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

        self.out_proj = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Conv1d(hidden_dim * 2, output_dim, kernel_size=1),
        )

    def forward(self, xs: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the ASTP layer.

        Args:
            xs (Tensor): input tensor, shape (B, C, T)
            x_lens (Tensor): length of the input sequences, shape (B,)

        Returns:
            Tensor: output tensor, shape (B, 2C, 1)
        """

        xs = self.inp_proj(xs)

        mask = make_padding_mask(x_lens)
        mask = mask.unsqueeze(1)

        xs = xs.masked_fill(mask, 0.0)
        x_lens = x_lens.unsqueeze(1)

        w = torch.logical_not(mask) / x_lens.unsqueeze(2)
        mean, std = self._compute_statistics(xs, w, dim=2)

        mean = mean.expand_as(xs)
        std = std.expand_as(xs)

        xs = torch.cat((xs, mean, std), dim=1)
        xs = self.network(xs)

        w = torch.softmax(xs.masked_fill(mask, float("-inf")), dim=2)
        mean, std = self._compute_statistics(xs, w, dim=2)

        xs = torch.cat([mean, std], dim=1)
        xs = self.out_proj(xs)

        return xs

    def _compute_statistics(self, xs: torch.Tensor, w: torch.Tensor, dim: int):
        mean = (w * xs).sum(dim, keepdim=True)
        var = (w * (xs - mean).pow(2)).sum(dim, keepdim=True)
        std = var.clamp(min=1e-9).sqrt()
        return mean, std
