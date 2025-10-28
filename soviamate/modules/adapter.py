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

"""Speaker adaptation module for multi-speaker audio synthesis."""

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as TAudio

from soviamate.layers.res2former import (
    AttentiveStatisticsPooling,
    LightweightSimpleTransformer,
    TimeFrequencyAdaptiveFusion,
)
from soviamate.utils.helper import make_padding_mask


class SpeakerAdapter(nn.Module):
    """Complete speaker adaptation module with integrated encoding and conditioning.

    Supports three conditioning strategies:
        - "global": Utterance-level fusion only (FiLM affine modulation, fastest)
        - "local": Frame-level fusion only (cross-attention temporal adaptation)
        - "hybrid": Combined utterance + frame fusion (most expressive, default)

    Args:
        input_dim: Dimension of input features to adapt (from codec).
        sample_rate: Audio sample rate in Hz. Default: 16000.
        n_fft: FFT size for spectrogram. Default: 512.
        win_length: STFT window length. Default: 400.
        hop_length: STFT hop length. Default: 160.
        num_mels: Number of mel frequency bins. Default: 80.
        encoder_hidden_dim: Hidden dimension of encoder. Default: 256.
        encoder_ffn_expansion: FFN expansion factor in encoder. Default: 4.
        encoder_blocks_per_stage: Number of LST blocks per stage. Default: (2, 2, 2, 2).
        encoder_kernel_sizes: Kernel sizes for encoder convolutions. Default: (5, 9, 11, 11).
        fusion_ffn_dim: FFN hidden dimension for fusion refinement. Default: 1024.
        fusion_num_heads: Number of attention heads for fusion. Default: 4.
        fusion_strategy: Fusion strategy ("global", "local", "hybrid"). Default: "hybrid".
        dropout: Dropout probability for all layers. Default: 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        num_mels: int = 80,
        encoder_hidden_dim: int = 256,
        encoder_ffn_expansion: int = 4,
        encoder_blocks_per_stage: List[int] = (2, 2, 2, 2),
        encoder_kernel_sizes: List[int] = (5, 9, 11, 11),
        fusion_ffn_dim: int = 1024,
        fusion_num_heads: int = 4,
        fusion_strategy: Literal["global", "local", "hybrid"] = "hybrid",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hop_length = hop_length
        self.fusion_strategy = fusion_strategy

        # ============================================================
        # Encoding Components (Res2Former architecture)
        # ============================================================

        # Mel-spectrogram frontend
        self.mel_transform = TAudio.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=num_mels,
        )

        # Input projection: num_mels -> encoder_hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(num_mels, encoder_hidden_dim),
            nn.LayerNorm(encoder_hidden_dim),
        )

        # Build multi-stage architecture with LST blocks
        self.encoder_stages = nn.ModuleList()
        for blocks_in_stage in encoder_blocks_per_stage:
            stage_blocks = nn.ModuleList(
                [
                    LightweightSimpleTransformer(
                        channels=encoder_hidden_dim,
                        kernel_sizes=encoder_kernel_sizes,
                        ffn_expansion=encoder_ffn_expansion,
                        dropout=dropout,
                    )
                    for _ in range(blocks_in_stage)
                ]
            )
            self.encoder_stages.append(stage_blocks)

        # Time-frequency adaptive fusion modules
        self.taff_modules = nn.ModuleList(
            [
                TimeFrequencyAdaptiveFusion(encoder_hidden_dim)
                for _ in range(len(encoder_blocks_per_stage))
            ]
        )

        # Encoder output head
        self.encoder_head = nn.Sequential(
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
            nn.LayerNorm(encoder_hidden_dim),
        )

        # Attentive pooling for utterance-level embeddings
        self.attentive_pooling = AttentiveStatisticsPooling(
            encoder_hidden_dim, input_dim
        )

        # Frame projection to output dimension
        self.frame_projection = nn.Sequential(
            nn.Linear(encoder_hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

        # ============================================================
        # Speaker Fusion Components
        # ============================================================

        # Utterance-level fusion (FiLM affine transformation)
        if fusion_strategy in ["global", "hybrid"]:
            self.film_projection = nn.Linear(input_dim, input_dim * 2)
            nn.init.zeros_(self.film_projection.weight)
            nn.init.zeros_(self.film_projection.bias)

        # Frame-level fusion (cross-attention temporal alignment)
        if fusion_strategy in ["local", "hybrid"]:
            self.norm_attn = nn.LayerNorm(input_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=fusion_num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_dropout = nn.Dropout(dropout)

        # Feature refinement (applied to all strategies)
        self.norm_ffn = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, fusion_ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_ffn_dim, input_dim),
            nn.Dropout(dropout),
        )

    def encode_speaker(
        self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale speaker embeddings from audio.

        Args:
            waveforms: Input audio of shape (B, T, 1) where T is audio samples.
            lengths: Actual lengths of waveforms of shape (B,), optional.

        Returns:
            utterance_embeddings: Utterance-level speaker vectors of shape (B, D).
            frame_embeddings: Frame-level speaker features of shape (B, T', D).
            frame_lengths: Lengths of frame embeddings of shape (B,).
        """
        # Compute mel-spectrogram: (B, T) -> (B, T', n_mels)
        mel_spec = self.mel_transform(waveforms.squeeze(-1))
        mel_spec = torch.log(mel_spec.clamp(min=1e-10) + 1e-6)

        # Compute frame lengths if provided
        if lengths is not None:
            frame_lengths = (lengths // self.hop_length + 1).long()
        else:
            batch_size, seq_len = mel_spec.size(0), mel_spec.size(1)
            frame_lengths = torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=mel_spec.device
            )

        # Apply input projection
        x = mel_spec.transpose(1, 2)
        x = self.input_proj(x).transpose(1, 2)

        # Forward through encoder stages with TAFF fusion
        taff_outputs = []
        for idx, stage_blocks in enumerate(self.encoder_stages):
            # Store stage input for fusion
            stage_input = x

            # Apply LST blocks within stage
            for lst_block in stage_blocks:
                x = lst_block(x)

            # Fuse stage output with its input
            fused_output = self.taff_modules[idx](x, stage_input)
            taff_outputs.append(fused_output)

            # Pass fused output to next stage
            x = fused_output

        # Aggregate all stage outputs
        stage_sum = sum(taff_outputs)

        # Apply encoder head and transpose
        x = stage_sum.transpose(1, 2)
        x = self.encoder_head(x)

        # Extract utterance-level and frame-level embeddings
        utterance_embeddings = self.attentive_pooling(x, frame_lengths)
        frame_embeddings = self.frame_projection(x)

        return utterance_embeddings, frame_embeddings, frame_lengths

    def apply_conditioning(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        utterance_embeddings: torch.Tensor,
        frame_embeddings: torch.Tensor,
        frame_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply speaker conditioning based on configured strategy.

        Args:
            x: Input features of shape (B, T, D).
            x_lengths: Actual lengths of input features of shape (B,).
            utterance_embeddings: Utterance-level speaker vectors of shape (B, D).
            frame_embeddings: Frame-level speaker features of shape (B, T', D).
            frame_lengths: Actual lengths of frame embeddings of shape (B,).

        Returns:
            conditioned_features: Speaker-adapted features of shape (B, T, D).
            feature_lengths: Lengths of features of shape (B,).
        """
        # Create padding masks for input and speaker frames
        query_mask = make_padding_mask(x_lengths)
        key_mask = make_padding_mask(frame_lengths)

        # ============================================================
        # Utterance-level fusion (FiLM affine transformation)
        # ============================================================

        if self.fusion_strategy in ["global", "hybrid"]:
            # Project to scale and shift parameters
            gamma, beta = torch.chunk(
                self.film_projection(utterance_embeddings), chunks=2, dim=-1
            )

            # Apply affine transformation: (1 + γ) * x + β
            x = (1.0 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)

        # ============================================================
        # Frame-level fusion (cross-attention temporal alignment)
        # ============================================================

        if self.fusion_strategy in ["local", "hybrid"]:
            # Pre-normalization
            attn_input = self.norm_attn(x)

            # Attend to frame-level speaker features
            attn_output, _ = self.cross_attn(
                query=attn_input,
                key=frame_embeddings,
                value=frame_embeddings,
                key_padding_mask=key_mask,
                need_weights=False,
            )

            attn_output = self.attn_dropout(attn_output)

            # Mask padding before residual
            if query_mask is not None:
                attn_output = attn_output.masked_fill(query_mask.unsqueeze(-1), 0.0)

            x = x + attn_output

        # ============================================================
        # Feature refinement (applied to all strategies)
        # ============================================================

        ffn_input = self.norm_ffn(x)
        ffn_output = self.ffn(ffn_input)

        # Mask padding positions before residual
        if query_mask is not None:
            ffn_output = ffn_output.masked_fill(query_mask.unsqueeze(-1), 0.0)

        # Residual connection
        x = x + ffn_output

        return x, x_lengths

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        speaker_prompts: torch.Tensor,
        speaker_prompt_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply speaker adaptation to input features.

        Args:
            x: Input features to adapt of shape (B, T, D).
            x_lengths: Actual lengths of input features of shape (B,).
            speaker_prompts: Speaker audio prompts of shape (B, T_prompt, 1).
            speaker_prompt_lengths: Actual lengths of prompts of shape (B,).

        Returns:
            outputs: Speaker-adapted features of shape (B, T, D).
            output_lengths: Lengths of features of shape (B,).
        """
        # Extract hierarchical speaker features from prompts
        utterance_embeddings, frame_embeddings, frame_lengths = self.encode_speaker(
            speaker_prompts, speaker_prompt_lengths
        )

        # Condition acoustic features with speaker identity
        outputs, output_lengths = self.apply_conditioning(
            x, x_lengths, utterance_embeddings, frame_embeddings, frame_lengths
        )

        return outputs, output_lengths
