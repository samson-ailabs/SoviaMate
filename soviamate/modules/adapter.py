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

"""Speaker feature extraction module for multi-speaker synthesis."""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio


class SpeakerAdapter(nn.Module):
    """SSL-based speaker adapter using WavLM for speaker feature extraction.

    Args:
        model_name (Literal["WAVLM_LARGE", "WAVLM_BASE"]): Pre-trained model identifier.
        extract_layer (int): Which WavLM layer to extract features from.
        output_dim (int): Output dimension for speaker embedding.
        dropout (float): Dropout probability for regularization.
    """

    def __init__(
        self,
        model_name: Literal["WAVLM_LARGE", "WAVLM_BASE"],
        extract_layer: int = 4,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.extract_layer = extract_layer

        # Load pre-trained WavLM model
        bundle = getattr(torchaudio.pipelines, model_name)
        self.feature_extractor = bundle.get_model()

        # Freeze pre-trained model
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Determine feature dimension
        feature_dim = 1024 if model_name == "WAVLM_LARGE" else 768

        # Adapt WavLM features to speaker embedding
        self.postnet = nn.Sequential(
            nn.Conv1d(feature_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(512, output_dim, kernel_size=1),
        )

    def forward(
        self, prompts: torch.Tensor, prompt_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract speaker features from audio using SSL.

        Args:
            prompts (Tensor): Audio waveform of shape `(B, T, 1)` or `(B, T)`.
            prompt_lengths (Optional[Tensor]): Actual lengths of shape `(B,)`.

        Returns:
            Tensor: Processed speaker features of shape `(B, T', D)`.
            Tensor: Frame-level embedding lengths of shape `(B,)`.
        """
        if prompts.dim() == 3:
            prompts = prompts.squeeze(-1)

        # Extract features from specified WavLM layer
        with torch.no_grad():
            features, _ = self.feature_extractor.extract_features(
                prompts, num_layers=self.extract_layer
            )
            extracted = features[self.extract_layer - 1]  # (B, T, D)

        # Adapt the rich WavLM features
        x = extracted.transpose(1, 2)
        outputs = self.postnet(x).transpose(1, 2)

        # Compute frame lengths based on actual processed output
        if prompt_lengths is not None:
            max_length = outputs.size(1)
            ratio = prompts.size(1) / max_length

            output_lengths = torch.ceil(prompt_lengths / ratio).long()
            output_lengths = torch.clamp(output_lengths, max=max_length)
        else:
            batch_size, max_length, _ = outputs.size()
            output_lengths = torch.full(
                (batch_size,), max_length, dtype=torch.long, device=outputs.device
            )

        return outputs, output_lengths
