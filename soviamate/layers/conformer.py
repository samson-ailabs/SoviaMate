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

"""Unified Streaming and Non-Streaming Conformer"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    r"""Self-attention module with streaming cache support.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads.
        dropout (float): dropout probability.
        norm_affine (bool): whether LayerNorm has learnable affine params.
    """

    def __init__(
        self, input_dim: int, num_heads: int, dropout: float, norm_affine: bool = True
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim, elementwise_affine=norm_affine)
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        contexts: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            inputs (Tensor): query input with shape `(B, T, D)`.
            padding_mask (Tensor): padding mask for inputs with shape `(B, T)`.
            attention_mask (Tensor): causal mask with shape `(T, T + left_context)`.
            contexts (Tensor): streaming cache with shape `(B, left_context, D)`.
            gamma (Tensor, optional): AdaLN scale with shape `(B, 1, D)`.
            beta (Tensor, optional): AdaLN shift with shape `(B, 1, D)`.

        Returns:
            Tuple[Tensor, Tensor]: (output, cache) with shapes `(B, T, D)`
                and `(B, left_context, D)`.
        """
        query = self.layer_norm(inputs)
        if gamma is not None and beta is not None:
            query = query * (1 + gamma) + beta

        key = value = torch.cat([contexts, query], dim=1)
        cache = key[:, query.size(1) :, :]

        num_pads = attention_mask.size(1) - padding_mask.size(1)
        kv_mask = F.pad(padding_mask, (num_pads, 0))

        x, _ = self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=kv_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = self.dropout(x)

        return x, cache


class ConvolutionModule(nn.Module):
    r"""Causal convolution module with streaming cache support.

    Args:
        input_dim (int): input dimension.
        kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float): dropout probability.
        norm_affine (bool): whether LayerNorm has learnable affine params.
    """

    def __init__(
        self, input_dim: int, kernel_size: int, dropout: float, norm_affine: bool = True
    ) -> None:
        super().__init__()

        if (kernel_size - 1) % 2 != 0:
            raise ValueError("kernel_size must be odd to achieve 'SAME' padding.")

        self.left_context = kernel_size - 1

        self.layer_norm = nn.LayerNorm(input_dim, elementwise_affine=norm_affine)
        self.pointwise_conv1 = nn.Conv1d(input_dim, input_dim, 1)
        self.activation1 = nn.GELU()
        self.depthwise_conv = nn.Conv1d(
            input_dim, input_dim, kernel_size, groups=input_dim, bias=False
        )

        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.activation2 = nn.GELU()
        self.pointwise_conv2 = nn.Conv1d(input_dim, input_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        padding_masks: torch.Tensor,
        contexts: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            inputs (Tensor): with shape `(B, T, D)`.
            padding_masks (Tensor): with shape `(B, T)`.
                A ``True`` value indicates the corresponding value will be ignored.
            contexts (Tensor): with shape `(B, D, kernel_size - 1)`.
                The cached left context used in the streaming convolution mechanism.
            gamma (Tensor, optional): AdaLN scale with shape `(B, 1, D)`.
            beta (Tensor, optional): AdaLN shift with shape `(B, 1, D)`.

        Returns:
            Tuple[Tensor, Tensor]: (output, cache) with shapes `(B, T, D)`
                and `(B, D, kernel_size - 1)`.
        """

        x = self.layer_norm(inputs)
        if gamma is not None and beta is not None:
            x = x * (1 + gamma) + beta

        x = self.activation1(self.pointwise_conv1(x.transpose(1, 2)))
        x = x.masked_fill(padding_masks.unsqueeze(1), 0.0)

        if contexts.size(2) > 0:
            x = torch.cat([contexts, x], dim=2)
            cache = x[:, :, -self.left_context :]
        else:
            x = F.pad(x, (self.left_context, 0))
            cache = contexts

        x = self.activation2(self.batch_norm(self.depthwise_conv(x)))
        x = self.dropout(self.pointwise_conv2(x)).transpose(1, 2)

        return x, cache


class FeedForwardModule(nn.Module):
    r"""Feed forward module.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float): dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            inputs (Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: outputs, with shape `(*, D)`.
        """
        return self.network(inputs)


class ConformerLayer(nn.Module):
    r"""The layer that constitutes Conformer model.

    Supports optional AdaLN-Gaussian conditioning when speaker_dim > 0.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_heads (int): number of attention heads.
        kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float): dropout probability.
        speaker_dim (int): speaker embedding dimension for AdaLN-Gaussian.
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_heads: int,
        kernel_size: int,
        dropout: float,
        speaker_dim: int = 0,
    ) -> None:
        super().__init__()

        self.ffn1_module = FeedForwardModule(input_dim, ffn_dim, dropout)
        self.attn_module = SelfAttentionModule(
            input_dim, num_heads, dropout, norm_affine=(speaker_dim == 0)
        )
        self.conv_module = ConvolutionModule(
            input_dim, kernel_size, dropout, norm_affine=(speaker_dim == 0)
        )
        self.ffn2_module = FeedForwardModule(input_dim, ffn_dim, dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

        if speaker_dim > 0:
            self.adaln_proj = nn.Linear(speaker_dim, 6 * input_dim)
            nn.init.normal_(self.adaln_proj.weight, std=1e-3)
            nn.init.zeros_(self.adaln_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        conv_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        conv_cache: torch.Tensor,
        attn_cache: torch.Tensor,
        spk_emb: Optional[torch.Tensor] = None,
    ):
        r"""
        Args:
            x (Tensor): input, with shape `(B, T, D)`.
            conv_mask (Tensor): convolution mask, with shape `(B, T)`.
            attn_mask (Tensor): attention mask, with shape `(T, T + left_context)`.
            conv_cache (Tensor): convolution cache, with shape `(B, D, kernel_size - 1)`.
            attn_cache (Tensor): attention cache, with shape `(B, left_context, D)`.
            spk_emb (Tensor, optional): speaker embedding `(B, S)`.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (output, conv_cache, attn_cache) with
                shapes `(B, T, D)`, `(B, D, kernel_size - 1)`, `(B, left_context, D)`.
        """
        gamma_attn = beta_attn = gate_attn = gamma_conv = beta_conv = gate_conv = None
        if spk_emb is not None and hasattr(self, "adaln_proj"):
            gamma_attn, beta_attn, gate_attn, gamma_conv, beta_conv, gate_conv = (
                self.adaln_proj(spk_emb).unsqueeze(1).chunk(6, dim=-1)
            )

        residual = x
        x = self.ffn1_module(x)
        x = x * 0.5 + residual

        residual = x
        x, attn_cache = self.attn_module(
            x, conv_mask, attn_mask, attn_cache, gamma_attn, beta_attn
        )
        x = gate_attn * x + residual if gate_attn is not None else x + residual

        residual = x
        x, conv_cache = self.conv_module(
            x, conv_mask, conv_cache, gamma_conv, beta_conv
        )
        x = gate_conv * x + residual if gate_conv is not None else x + residual

        residual = x
        x = self.ffn2_module(x)
        x = x * 0.5 + residual

        x = self.layer_norm(x)

        return x, conv_cache, attn_cache
