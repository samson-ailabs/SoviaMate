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

"""Adapter modules for extracting conditioning features."""

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from soviamate.utils.helper import make_padding_mask


class SEConvBlock(nn.Module):
    """Squeeze-Excitation convolutional block with masked channel attention.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size. Defaults to 3.
        se_reduction (int): SE bottleneck reduction ratio. Defaults to 8.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) // 2

        # Main branch
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # SE branch
        se_channels = max(out_channels // se_reduction, 8)
        self.se_fc1 = nn.Conv1d(out_channels, se_channels, 1)
        self.se_fc2 = nn.Conv1d(se_channels, out_channels, 1)

        # Residual projection
        self.proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with SE channel attention.

        Args:
            x (Tensor): Input of shape (B, C, T).
            mask (Tensor): Mask of shape (B, 1, T), 1=valid, 0=pad.

        Returns:
            Tensor: Output of shape (B, C_out, T).
        """
        residual = self.proj(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        masked_sum = (out * mask).sum(dim=-1, keepdim=True)
        valid_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)

        se = self.se_fc1(masked_sum / valid_count)
        se = torch.sigmoid(self.se_fc2(F.relu(se)))

        return out * se + residual


class SpeakerAdapter(nn.Module):
    """Speaker adapter that extracts frame-level speaker features from audio.

    Args:
        output_dim (int): Output dimension for decoder cross-attention.
        num_layers (int): Number of WavLM layers to use. Defaults to 6.
        model_name (str): Pre-trained WavLM model identifier. Defaults to "WAVLM_LARGE".
        postnet_dims (tuple): Hidden dimensions for postnet blocks. Defaults to (128, 256, 512).
    """

    def __init__(
        self,
        output_dim: int,
        num_layers: int = 6,
        model_name: Literal["WAVLM_LARGE", "WAVLM_BASE"] = "WAVLM_LARGE",
        postnet_dims: Tuple[int, ...] = (128, 256, 512),
    ):
        super().__init__()

        self.num_layers = num_layers
        self.output_dim = output_dim

        bundle = getattr(torchaudio.pipelines, model_name)
        self.feature_extractor = bundle.get_model()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        feature_dim = 1024 if model_name == "WAVLM_LARGE" else 768
        channel_dims = [feature_dim] + list(postnet_dims) + [output_dim]

        self.postnet = nn.ModuleList(
            [
                SEConvBlock(channel_dims[i], channel_dims[i + 1])
                for i in range(len(channel_dims) - 1)
            ]
        )

    def forward(
        self, prompts: torch.Tensor, prompt_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract frame-level speaker features from audio.

        Args:
            prompts (Tensor): Audio waveform of shape (B, T) or (B, T, 1).
            prompt_lengths (Tensor): Actual lengths in samples of shape (B,).

        Returns:
            Tensor: Speaker features of shape (B, T', D).
            Tensor: Feature lengths of shape (B,).
        """
        if prompts.dim() == 3:
            prompts = prompts.squeeze(-1)

        # Extract SSL features
        with torch.no_grad():
            features, _ = self.feature_extractor.extract_features(
                prompts, num_layers=self.num_layers
            )

        # Weighted sum of layer features
        stacked = torch.stack(features, dim=1).transpose(2, 3)
        weights = F.softmax(self.layer_weights, dim=0)
        features = torch.einsum("bldt,l->bdt", stacked, weights)

        # Compute feature lengths
        ratio = prompts.size(1) / features.size(2)
        lengths = torch.ceil(prompt_lengths / ratio).long()
        lengths = torch.clamp(lengths, max=features.size(2))

        # Apply postnet with valid mask
        mask = ~make_padding_mask(lengths).unsqueeze(1)
        for block in self.postnet:
            features = block(features, mask)

        return features.transpose(1, 2), lengths
