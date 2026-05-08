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
import torchaudio

from soviamate.layers.campplus import CAMPPlus
from soviamate.layers.eres2netv2 import ERes2NetV2
from soviamate.layers.processor import KaldiFbank
from soviamate.utils.helper import make_padding_mask


class _SelfAttentionBlock(nn.Module):
    """Position-agnostic self-attention block.

    Args:
        d_model (int): Feature dimension.
        num_heads (int): Number of attention heads.
        kernel_size (int): Kernel size for the Conv1d layer.
        dropout (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(
        self, d_model: int, num_heads: int, kernel_size: int, dropout: float = 0.1
    ):
        super().__init__()
        self.norm = nn.GroupNorm(32, d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Apply self-attention and convolution with residual connection.

        Args:
            x (Tensor): Input features of shape (B, T, D).
            padding_mask (Tensor, optional): Key padding mask of shape (B, T).

        Returns:
            Tensor: Refined features of shape (B, T, D).
        """
        residual = x

        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.self_attn(
            x, x, x, key_padding_mask=padding_mask, need_weights=False
        )
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)

        return x + residual


class _MemoryAugmentedModule(nn.Module):
    """Memory-augmented timbre modeling module.

    Args:
        input_dim (int): Input feature dimension.
        attention_dim (int): Internal dimension for self-attention blocks.
        output_dim (int): Output feature dimension.
        num_heads (int): Number of attention heads. Defaults to 4.
        num_blocks (int): Number of self-attention blocks. Defaults to 2.
        kernel_size (int): Kernel size for Conv1d layers. Defaults to 5.
        dropout (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
        output_dim: int | None = None,
        num_heads: int = 4,
        num_blocks: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.input_projection = nn.Conv1d(
            input_dim, attention_dim, kernel_size, padding=padding
        )

        self.blocks = nn.ModuleList(
            [
                _SelfAttentionBlock(attention_dim, num_heads, kernel_size, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.film_scale = nn.Parameter(torch.ones(attention_dim))
        self.film_bias = nn.Parameter(torch.zeros(attention_dim))

        if output_dim is not None and output_dim != attention_dim:
            self.output_projection = nn.Linear(attention_dim, output_dim)
        else:
            self.output_projection = None

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project, refine through SA blocks, and apply FiLM.

        Args:
            features (Tensor): Input features of shape (B, C, T).
            lengths (Tensor): Valid frame lengths of shape (B,).

        Returns:
            Tensor: Refined features of shape (B, T, D).
            Tensor: Feature lengths of shape (B,).
        """
        features = self.input_projection(features).transpose(1, 2)
        padding_mask = make_padding_mask(lengths)

        for block in self.blocks:
            features = block(features, padding_mask)

        features = self.film_scale * features + self.film_bias

        if self.output_projection is not None:
            features = self.output_projection(features)

        features = features.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return features, lengths


class MemoryAugmentedAdapter(nn.Module):
    """Speaker adapter combining mel features with a frozen CAM++ speaker
    embedding, refined through memory-augmented self-attention.

    Args:
        output_dim (int): Output dimension for decoder cross-attention.
        attention_dim (int): Internal dimension for self-attention blocks. Defaults to 512.
        n_mels (int): Number of mel filterbank channels. Defaults to 80.
        sample_rate (int): Audio sample rate in Hz. Defaults to 16000.
        sv_checkpoint (str): Path to pretrained CAM++ checkpoint.
        sv_embedding_size (int): CAM++ output embedding dimension. Defaults to 192.
        num_heads (int): Number of attention heads. Defaults to 8.
        num_blocks (int): Number of self-attention blocks. Defaults to 4.
        kernel_size (int): Kernel size for Conv1d layers. Defaults to 5.
        dropout (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(
        self,
        output_dim: int,
        attention_dim: int = 512,
        n_mels: int = 80,
        sample_rate: int = 16000,
        sv_checkpoint: str = "",
        sv_embedding_size: int = 192,
        num_heads: int = 8,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.win_length = int(25.0 / 1000.0 * sample_rate)  # 25ms frame length
        self.hop_length = int(10.0 / 1000.0 * sample_rate)  # 10ms frame shift

        # Mel spectrogram for computing mel from waveform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=n_mels,
        )

        # Frozen CAM++ for global speaker embedding
        self.sv_model = CAMPPlus(feat_dim=n_mels, embedding_size=sv_embedding_size)

        if sv_checkpoint:
            state_dict = torch.load(
                sv_checkpoint, map_location="cpu", weights_only=True
            )
            self.sv_model.load_state_dict(state_dict)

        self.sv_model.eval()
        self.sv_model.requires_grad_(False)

        # Memory-augmented timbre refinement
        self.voice_print = _MemoryAugmentedModule(
            input_dim=n_mels + sv_embedding_size,
            attention_dim=attention_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def train(self, mode: bool = True) -> "MemoryAugmentedAdapter":
        """Override to keep sv_model in eval mode (frozen BatchNorm)."""
        super().train(mode)
        self.sv_model.eval()
        return self

    def forward(
        self,
        prompt_audios: torch.Tensor,
        prompt_audio_lengths: torch.Tensor,
        prompt_fbanks: torch.Tensor,
        prompt_fbank_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract frame-level speaker features from audio.

        Args:
            prompt_audios (Tensor): Prompt waveform of shape (B, T, 1).
            prompt_audio_lengths (Tensor): Actual lengths in samples of shape (B,).
            prompt_fbanks (Tensor): Precomputed fbank of shape (B, T', n_mels).
            prompt_fbank_lengths (Tensor): Fbank feature lengths of shape (B,).

        Returns:
            Tensor: Speaker features of shape (B, T', D).
            Tensor: Feature lengths of shape (B,).
        """
        if prompt_audios.dim() == 3:
            prompt_audios = prompt_audios.squeeze(-1)

        # SV path: extract speaker embedding from fbank
        with torch.no_grad():
            spk_emb = self.sv_model(prompt_fbanks.transpose(1, 2), prompt_fbank_lengths)

        # Mel path: compute from prompt waveform
        mel_spec = self.mel_spectrogram(prompt_audios)
        mel_spec = torch.log(mel_spec.clamp(min=1e-5))

        # Combine mel + SV → timbre module
        sv_expanded = spk_emb.unsqueeze(2).expand(-1, -1, mel_spec.size(2))
        features = torch.cat([mel_spec, sv_expanded], dim=1)

        feature_lengths = prompt_audio_lengths // self.hop_length + 1
        feature_lengths = torch.clamp(feature_lengths, min=1, max=mel_spec.size(2))

        # Memory-augmented refinement to produce final speaker features
        outputs, output_lengths = self.voice_print(features, feature_lengths)

        return outputs, output_lengths


class GlobalSpeakerAdapter(nn.Module):
    """Extract utterance-level speaker embedding directly from raw audio.

    Args:
        sample_rate (int): Audio sample rate in Hz. Defaults to ``16000``.
        n_mels (int): Number of mel filterbank channels. Defaults to ``80``.
        encoder_type (Literal["campplus", "eres2net"]): Speaker encoder type.
        sv_checkpoint (str): Path to the pretrained speaker encoder checkpoint.
        sv_embedding_size (int): Output embedding dimension. Defaults to ``192``.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        encoder_type: Literal["campplus", "eres2net"] = "campplus",
        sv_checkpoint: str = "",
        sv_embedding_size: int = 192,
    ):
        super().__init__()

        self.embedding_size = sv_embedding_size
        self.fbank = KaldiFbank(sample_rate=sample_rate, n_mels=n_mels)

        if encoder_type == "eres2net":
            self.sv_model = ERes2NetV2(
                feat_dim=n_mels, embedding_size=sv_embedding_size
            )
        elif encoder_type == "campplus":
            self.sv_model = CAMPPlus(feat_dim=n_mels, embedding_size=sv_embedding_size)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        if sv_checkpoint:
            state_dict = torch.load(
                sv_checkpoint, map_location="cpu", weights_only=True
            )
            self.sv_model.load_state_dict(state_dict)

        self.sv_model.eval()
        self.sv_model.requires_grad_(False)

    def train(self, mode: bool = True) -> "GlobalSpeakerAdapter":
        """Override to keep sv_model in eval mode (frozen BatchNorm)."""
        super().train(mode)
        self.sv_model.eval()
        return self

    def forward(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Extract utterance-level speaker embedding from raw audio.

        Args:
            audios (Tensor): Audio waveforms, shape ``(B, T)`` or ``(B, T, 1)``.
            audio_lengths (Tensor): Per-sample valid sample counts, shape ``(B,)``.

        Returns:
            Tensor: Speaker embeddings, shape ``(B, sv_embedding_size)``.
        """
        features, feat_lengths = self.fbank(audios, audio_lengths)
        return self.sv_model(features, feat_lengths)
