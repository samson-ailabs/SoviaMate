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

"""Res2Former: Integrating Res2Net and Transformer for Speaker Verification

This module implements the Res2Former architecture as described in:
Chen et al., "Res2Former: Integrating Res2Net and Transformer for a Highly Efficient
Speaker Verification System", Electronics 2025, 14, 2489.

Key components:
- Lightweight Simple Transformer (LST) with Multi-Scale Convolutional Attention (MSCA)
- Global Response Normalization (GRN) based Feed-Forward Network
- Time-Frequency Adaptive Feature Fusion (TAFF) mechanism
- Multi-stage hierarchical architecture with Res2Net-inspired structure
- Attentive Statistics Pooling (ASP) for speaker embedding extraction

Architecture highlights:
- Reduces computational complexity while maintaining Transformer benefits
- Achieves state-of-the-art accuracy with significantly fewer parameters
- Suitable for resource-constrained environments (IoT, mobile devices)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConvolutionalAttention(nn.Module):
    """Multi-Scale Convolutional Attention (MSCA) module.

    Different kernel sizes mimic different attention heads while significantly
    reducing computational complexity compared to standard multi-head attention.

    Args:
        channels (int): Number of input/output channels.
        kernel_sizes (List[int]): List of kernel sizes for multi-scale convolution.
    """

    def __init__(self, channels: int, kernel_sizes: List[int]):
        super().__init__()

        # Split channels across branches
        num_branches = len(kernel_sizes)
        head_dim = channels // num_branches

        # Attention projections (one per branch)
        self.attn_projs = nn.ModuleList(
            [nn.Conv1d(channels, head_dim, kernel_size=1) for _ in range(num_branches)]
        )

        # Multi-scale value branches
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, head_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(
                        head_dim,
                        head_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=head_dim,
                    ),
                )
                for k in kernel_sizes
            ]
        )

        # Output projection
        self.proj_out = nn.Conv1d(num_branches * head_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MSCA.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Output tensor of shape (B, C, T)
        """
        # Multi-scale attention: Ai ⊙ Vi for each branch
        head_outs = []
        for attn_proj, branch in zip(self.attn_projs, self.branches):
            head_out = attn_proj(x) * branch(x)
            head_outs.append(head_out)

        # Concatenate heads, project, and add residual
        msca = torch.cat(head_outs, dim=1)
        out = x + self.proj_out(msca)

        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention module using MSCA.

    Args:
        channels (int): Number of input/output channels.
        kernel_sizes (List[int]): List of kernel sizes for MSCA heads.
    """

    def __init__(self, channels: int, kernel_sizes: List[int]):
        super().__init__()

        # Input projection
        self.proj_in = nn.Conv1d(channels, channels, kernel_size=1)
        self.gelu1 = nn.GELU()

        # Multi-scale convolutional attention
        self.msca = MultiScaleConvolutionalAttention(channels, kernel_sizes)

        # Output projection
        self.gelu2 = nn.GELU()
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of multi-head self-attention.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Output tensor of shape (B, C, T)
        """
        x = self.proj_in(x)
        x = self.gelu1(x)
        x = self.msca(x)
        x = self.gelu2(x)
        x = self.proj_out(x)
        return x


class GlobalResponseNormalization(nn.Module):
    """Global Response Normalization (GRN) module.

    GRN applies global response normalization on a per-sample basis, enabling
    flexible control of information flow and enhancing network expressiveness.

    Args:
        channels (int): Number of feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Learnable parameters for each feature channel
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GRN to input features.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Normalized and scaled tensor of shape (B, C, T)
        """
        # Compute global response for each time step across all channels
        global_response = torch.norm(x, p=2, dim=1, keepdim=True)

        # Normalize features by global response with numerical stability
        x_normalized = x / (global_response + 1e-6)

        # Apply learnable affine transformation
        return self.gamma * x_normalized + self.beta


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network with Global Response Normalization.

    Args:
        channels (int): Number of input/output channels.
        expansion (int, optional): Hidden dimension expansion ratio. Default: 4.
        dropout (float, optional): Dropout probability. Default: 0.0.
    """

    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()

        # Input projection and activation
        self.conv1 = nn.Conv1d(channels, channels * expansion, kernel_size=1)
        self.gelu = nn.GELU()

        # Global Response Normalization
        self.norm = GlobalResponseNormalization(channels * expansion)

        # Dropout and output projection
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels * expansion, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of GRN-based FFN.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Output tensor of shape (B, C, T)
        """
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class TimeFrequencyAdaptiveFusion(nn.Module):
    """Time-Frequency Adaptive Feature Fusion (TAFF) mechanism.

    Models time-frequency relationships through attention weighting, enabling
    fine-grained feature propagation. Constructs a weight matrix positively
    correlated with frequencies at each specific location.

    Args:
        channels (int): Number of feature channels.
        reduction (int, optional): Channel reduction ratio. Default: 8.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        reduced_channels = max(channels // reduction, 1)

        # Global pooling layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Equation 17: First convolution with BatchNorm and GELU
        self.conv1 = nn.Conv1d(channels, reduced_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(reduced_channels)
        self.gelu = nn.GELU()

        # Equation 18: Second convolution with BatchNorm
        self.conv2 = nn.Conv1d(reduced_channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(channels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply TAFF to fuse two feature tensors.

        Args:
            x: First input tensor of shape (B, C, T)
            y: Second input tensor of shape (B, C, T)

        Returns:
            Fused output tensor of shape (B, C, T)
        """
        # Equation 16: Global average pooling over time dimension
        combined = x + y
        pooled = self.avg_pool(combined)

        # Equation 17: First convolution with BatchNorm and GELU
        out = self.conv1(pooled)
        out = self.bn1(out)
        out = self.gelu(out)

        # Equation 18: Second convolution with BatchNorm
        out = self.conv2(out)
        out = self.bn2(out)

        # Equation 19: Softmax over channel dimension
        att = self.softmax(out)

        # Equation 20: Apply attention weights and fuse
        out = (x * att) + (y * att)

        return out


class LightweightSimpleTransformer(nn.Module):
    """Lightweight Simple Transformer (LST) block.

    Core building block that combines MSCA-based Multi-Head Self-Attention and
    GRN-based FFN with residual connections. Replaces expensive dot-product self-attention
    with efficient multi-scale convolutional attention while maintaining Transformer benefits.

    Args:
        channels (int): Number of feature channels.
        kernel_sizes (List[int]): Kernel sizes for MSCA.
        ffn_expansion (int, optional): FFN hidden dimension expansion. Default: 4.
        dropout (float, optional): Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int],
        ffn_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.mhsa = MultiHeadSelfAttention(channels, kernel_sizes)

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = FeedForwardNetwork(channels, ffn_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of LST block.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Output tensor of shape (B, C, T)
        """
        # Multi-head self-attention with residual
        x_norm = x.transpose(1, 2)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.transpose(1, 2)
        x = x + self.mhsa(x_norm)

        # Feed-forward network with residual
        x_norm = x.transpose(1, 2)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.transpose(1, 2)
        x = x + self.ffn(x_norm)

        return x


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling (ASP) layer.

    Aggregates variable-length frame-level features into fixed-dimensional
    speaker embeddings using attention-weighted statistics.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Output embedding dimension.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        # Attention mechanism
        self.attn_linear = nn.Linear(input_dim, input_dim)
        self.attn_tanh = nn.Tanh()
        self.attn_vector = nn.Linear(input_dim, 1, bias=False)

        # Output projection (concatenates mean and std)
        self.output_linear = nn.Linear(input_dim * 2, output_dim)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention-weighted pooled features.

        Args:
            x: Input features of shape (B, T, D)
            lengths: Actual sequence lengths of shape (B,), optional

        Returns:
            Speaker embedding of shape (B, output_dim)
        """

        # Compute attention scores
        attn = self.attn_tanh(self.attn_linear(x))
        attn = self.attn_vector(attn).squeeze(-1)

        # Apply length masking if provided
        if lengths is not None:
            idxs = torch.arange(x.shape[1], device=x.device)
            mask = idxs[None, :] >= lengths[:, None]
            attn = attn.masked_fill(mask, -1e3)

        # Normalize attention scores and compute weighted mean
        attn = F.softmax(attn, dim=1).unsqueeze(-1)
        mean = torch.sum(attn * x, dim=1)

        # Compute weighted standard deviation
        var = torch.sum(attn * (x**2), dim=1) - mean**2
        std = torch.sqrt(var.clamp(min=1e-10))

        # Concatenate statistics and project to output embedding
        pooled = torch.cat([mean, std], dim=1)
        embedding = self.output_linear(pooled)

        return embedding


class Res2Former(nn.Module):
    """Res2Former: Multi-stage architecture with overall time-frequency adaptive fusion.

    Implements the Res2Former model with LST blocks and overall TAFF across stages.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Output embedding dimension.
        num_channels (int): Number of channels for all stages.
        stage_blocks (List[int]): Number of LST blocks per stage.
        kernel_sizes (List[int]): Kernel sizes for MSCA in each stage.
        ffn_expansion (int, optional): FFN expansion factor. Default: 4.
        dropout (float, optional): Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_channels: int,
        stage_blocks: List[int],
        kernel_sizes: List[int],
        ffn_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Input projection: input_dim -> num_channels
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, num_channels),
            nn.LayerNorm(num_channels),
        )

        # Build multi-stage architecture
        self.stages = nn.ModuleList()
        for num_blocks in stage_blocks:
            stage_layers = nn.ModuleList(
                [
                    LightweightSimpleTransformer(
                        channels=num_channels,
                        kernel_sizes=kernel_sizes,
                        ffn_expansion=ffn_expansion,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
            self.stages.append(stage_layers)

        # TAFF modules: fuses current stage with previous
        self.overall_taff = nn.ModuleList(
            [
                TimeFrequencyAdaptiveFusion(num_channels)
                for _ in range(len(stage_blocks))
            ]
        )

        # Final fusion: Linear projection + LayerNorm
        self.final_fusion = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.LayerNorm(num_channels),
        )

        # Attentive Statistics Pooling (Equation 15)
        self.asp = AttentiveStatisticsPooling(num_channels, output_dim)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with overall time-frequency adaptive fusion.

        Args:
            x: Input features of shape (B, T, D) where D=input_dim.
            lengths: Actual sequence lengths of shape (B,), optional.

        Returns:
            Speaker embeddings of shape (B, embedding_dim).
        """
        # Apply input projection
        x = self.input_proj(x)
        x = x.transpose(1, 2)

        # Forward through all stages with cascaded TAFF fusion
        taff_outputs = []

        for i, stage_layers in enumerate(self.stages):
            # Store input to current stage for TAFF fusion
            stage_input = x

            # Apply LST blocks for current stage
            for lst_block in stage_layers:
                x = lst_block(x)

            # TAFF fusion: current stage output with its input
            taff_output = self.overall_taff[i](x, stage_input)
            taff_outputs.append(taff_output)

            # Cascade: TAFF output becomes input to next stage
            x = taff_output

        # Sum all TAFF outputs for final aggregation
        fused_sum = sum(taff_outputs)

        # Apply Linear + LayerNorm
        x = fused_sum.transpose(1, 2)
        x = self.final_fusion(x)

        # Apply Attentive Statistics Pooling
        embedding = self.asp(x, lengths)

        return embedding


def res2former_base(input_dim: int = 80, output_dim: int = 192, **kwargs):
    """Res2Former Base model configuration.

    Args:
        input_dim: Input feature dimension (default: 80 for mel-spectrogram).
        output_dim: Output embedding dimension.
        **kwargs: Additional arguments for Res2Former.

    Returns:
        Res2Former model with base configuration.
    """
    return Res2Former(
        input_dim=input_dim,
        num_channels=192,
        stage_blocks=[2, 2, 2, 2],
        kernel_sizes=[5, 9, 11, 11],
        output_dim=output_dim,
        **kwargs,
    )


def res2former_large(input_dim: int = 80, output_dim: int = 192, **kwargs):
    """Res2Former Large model configuration.

    Args:
        input_dim: Input feature dimension (default: 80 for mel-spectrogram).
        output_dim: Output embedding dimension.
        **kwargs: Additional arguments for Res2Former.

    Returns:
        Res2Former model with large configuration.
    """
    return Res2Former(
        input_dim=input_dim,
        output_dim=output_dim,
        num_channels=256,
        stage_blocks=[2, 2, 2, 2],
        kernel_sizes=[5, 9, 11, 11],
        **kwargs,
    )
