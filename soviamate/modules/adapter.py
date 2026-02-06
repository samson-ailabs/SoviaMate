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


class SpeakerAdapter(nn.Module):
    """Speaker adapter that extracts frame-level speaker features from audio.

    Args:
        output_dim (int): Output dimension for decoder cross-attention.
        num_layers (int): Number of WavLM layers to use for fusion. Defaults to 6.
        model_name (str): Pre-trained WavLM model identifier. Defaults to "WAVLM_LARGE".
    """

    def __init__(
        self,
        output_dim: int,
        num_layers: int = 6,
        model_name: Literal["WAVLM_LARGE", "WAVLM_BASE"] = "WAVLM_LARGE",
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
        self.projection = nn.Linear(feature_dim, output_dim)

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

        # Extract SSL features from multiple layers
        with torch.no_grad():
            features, _ = self.feature_extractor.extract_features(
                prompts, num_layers=self.num_layers
            )

        # Multi-layer fusion with learned weights
        stacked = torch.stack(features, dim=1)
        weights = F.softmax(self.layer_weights, dim=0)

        features = torch.einsum("bltd,l->btd", stacked, weights)
        features = self.projection(features)

        # Compute feature lengths
        ratio = prompts.size(1) / features.size(1)
        lengths = torch.ceil(prompt_lengths / ratio).long()
        lengths = torch.clamp(lengths, max=features.size(1))

        return features, lengths
