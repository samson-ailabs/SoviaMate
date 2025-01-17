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

""" Unified Streaming and Non-Streaming Conformer """

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["ConformerLayer"]


class _SelfAttentionModule(nn.Module):
    r"""Chunk-based self attention module.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads.
        dropout (float): dropout probability.
    """

    def __init__(self, input_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        padding_masks: torch.Tensor,
        attention_masks: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            inputs (torch.Tensor): with shape `(B, T, D)`.
            padding_masks (torch.Tensor): with shape `(B, T)`.
                A ``True`` value indicates the corresponding key value will be ignored.
            attention_masks (torch.Tensor): with shape `(T, T + left_context)`.
                A ``True`` value indicates the corresponding position is not allowed to attend.
            contexts (torch.Tensor): with shape `(B, left_context, D)`.
                The cached left context used in the streaming self-attention mechanism.

        Returns:
            torch.Tensor: outputs, with shape `(B, T, D)`.
        """

        q = self.layer_norm(inputs)

        k = v = torch.cat([contexts, q], dim=1)
        cache = k[:, q.size(1) :, :]

        num_pads = attention_masks.size(1) - padding_masks.size(1)
        padding_masks = F.pad(padding_masks, (num_pads, 0), value=0)

        x, _ = self.attention(
            q, k, v, key_padding_mask=padding_masks, attn_mask=attention_masks
        )

        x = self.dropout(x)

        return x, cache


class _ConvolutionModule(nn.Module):
    r"""Causal convolution module.

    Args:
        input_dim (int): input dimension.
        kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float): dropout probability.
    """

    def __init__(self, input_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()

        if (kernel_size - 1) % 2 != 0:
            raise ValueError("kernel_size must be odd to achieve 'SAME' padding.")

        self.left_context = kernel_size - 1

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.pointwise_conv1 = nn.Conv1d(input_dim, 2 * input_dim, 1)
        self.activation1 = nn.GLU(dim=1)

        self.depthwise_conv = nn.Conv1d(
            input_dim, input_dim, kernel_size, groups=input_dim
        )

        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.activation2 = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(input_dim, input_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, inputs: torch.Tensor, padding_masks: torch.Tensor, contexts: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            inputs (torch.Tensor): with shape `(B, T, D)`.
            padding_masks (torch.Tensor): with shape `(B, T)`.
                A ``True`` value indicates the corresponding value will be ignored.
            contexts (torch.Tensor): with shape `(B, D, kernel_size - 1)`.
                The cached left context used in the streaming convolution mechanism.

        Returns:
            torch.Tensor: outputs, with shape `(B, T, D)`.
        """

        x = self.layer_norm1(inputs)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.activation1(x)

        mask = padding_masks.unsqueeze(1)
        x = x.masked_fill(mask, 0.0)

        x = torch.cat([contexts, x], dim=2)
        cache = x[:, :, -self.left_context :]
        x = self.depthwise_conv(x)

        x = x.transpose(1, 2)
        x = self.layer_norm2(x)
        x = x.transpose(1, 2)

        x = self.activation2(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)

        return x, cache


class _FeedForwardModule(nn.Module):
    r"""Feed forward module.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float): dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        self.sequential = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            inputs (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: outputs, with shape `(*, D)`.
        """
        return self.sequential(inputs)


class ConformerLayer(nn.Module):
    r"""The layer that constitutes Conformer model.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_heads (int): number of attention heads.
        kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_heads: int,
        kernel_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.ffn1_module = _FeedForwardModule(input_dim, ffn_dim, dropout)
        self.attn_module = _SelfAttentionModule(input_dim, num_heads, dropout)
        self.conv_module = _ConvolutionModule(input_dim, kernel_size, dropout)
        self.ffn2_module = _FeedForwardModule(input_dim, ffn_dim, dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        conv_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        conv_cache: torch.Tensor,
        attn_cache: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): input, with shape `(B, T, D)`.
            conv_mask (torch.Tensor): convolution mask, with shape `(B, T)`.
            attn_mask (torch.Tensor): attention mask, with shape `(T, T + left_context)`.
            conv_cache (torch.Tensor): convolution cache, with shape `(B, D, kernel_size - 1)`.
            attn_cache (torch.Tensor): attention cache, with shape `(B, left_context, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """

        residual = x
        x = self.ffn1_module(x)
        x = x * 0.5 + residual

        residual = x
        x, attn_cache = self.attn_module(x, conv_mask, attn_mask, attn_cache)
        x = x + residual

        residual = x
        x, conv_cache = self.conv_module(x, conv_mask, conv_cache)
        x = x + residual

        residual = x
        x = self.ffn2_module(x)
        x = x * 0.5 + residual

        x = self.layer_norm(x)

        return x, conv_cache, attn_cache
