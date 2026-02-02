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

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    r"""Chunk-based self attention module.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads.
        dropout (float): dropout probability.
        epsilon (float, optional): LayerNorm epsilon. (Default: 1e-5)
    """

    def __init__(
        self, input_dim: int, num_heads: int, dropout: float, epsilon: float = 1e-5
    ) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.layer_norm = nn.LayerNorm(input_dim, eps=epsilon)
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
            inputs (Tensor): with shape `(B, T, D)`.
            padding_masks (Tensor): with shape `(B, T)`.
                A ``True`` value indicates the corresponding key value will be ignored.
            attention_masks (Tensor): with shape `(T, T + left_context)`.
                A ``True`` value indicates the corresponding position is not allowed to attend.
            contexts (Tensor): with shape `(B, left_context, D)`.
                The cached left context used in the streaming self-attention mechanism.

        Returns:
            torch.Tensor: outputs, with shape `(B, T, D)`.
        """

        query = self.layer_norm(inputs)

        key = value = torch.cat([contexts, query], dim=1)
        cache = key[:, query.size(1) :, :]

        num_pads = attention_masks.size(1) - padding_masks.size(1)
        padding_masks = F.pad(padding_masks, (num_pads, 0), value=0)

        x, _ = self.attention(
            query,
            key,
            value,
            key_padding_mask=padding_masks,
            attn_mask=attention_masks,
            need_weights=False,
        )

        x = self.dropout(x)

        return x, cache


class CrossAttentionModule(nn.Module):
    r"""Position-agnostic cross-attention module for external conditioning.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads.
        attn_dim (int): dimension of cross-attention keys/values.
        dropout (float): dropout probability.
        epsilon (float, optional): LayerNorm epsilon. (Default: 1e-5)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        attn_dim: int,
        dropout: float,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()

        self.query_norm = nn.LayerNorm(input_dim, eps=epsilon)
        self.key_value_norm = nn.LayerNorm(attn_dim, eps=epsilon)

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            kdim=attn_dim,
            vdim=attn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        prompts: torch.Tensor,
        padding_masks: torch.Tensor = None,
        prompt_masks: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""Apply position-agnostic cross-attention.

        Args:
            inputs (Tensor): input tensor with shape `(B, T, D)`.
            prompts (Tensor): prompt features with shape `(B, T', D')`.
            padding_masks (torch.Tensor, optional): padding mask for inputs with shape `(B, T)`.
            prompt_masks (torch.Tensor, optional): padding mask for prompts with shape `(B, T')`.

        Returns:
            torch.Tensor: output tensor with shape `(B, T, D)`.
        """
        query = self.query_norm(inputs)
        key_value = self.key_value_norm(prompts)

        x, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=prompt_masks,
            need_weights=False,
        )

        x = self.dropout(x)

        if padding_masks is not None:
            x = x.masked_fill(padding_masks.unsqueeze(-1), 0.0)

        return x


class ConvolutionModule(nn.Module):
    r"""Causal convolution module.

    Args:
        input_dim (int): input dimension.
        kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float): dropout probability.
        epsilon (float, optional): LayerNorm epsilon. (Default: 1e-5)
    """

    def __init__(
        self, input_dim: int, kernel_size: int, dropout: float, epsilon: float = 1e-5
    ) -> None:
        super().__init__()

        if (kernel_size - 1) % 2 != 0:
            raise ValueError("kernel_size must be odd to achieve 'SAME' padding.")

        self.left_context = kernel_size - 1

        self.layer_norm = nn.LayerNorm(input_dim, eps=epsilon)
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
        self, inputs: torch.Tensor, padding_masks: torch.Tensor, contexts: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            inputs (Tensor): with shape `(B, T, D)`.
            padding_masks (Tensor): with shape `(B, T)`.
                A ``True`` value indicates the corresponding value will be ignored.
            contexts (Tensor): with shape `(B, D, kernel_size - 1)`.
                The cached left context used in the streaming convolution mechanism.

        Returns:
            torch.Tensor: outputs, with shape `(B, T, D)`.
        """

        x = self.layer_norm(inputs)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.activation1(x)

        mask = padding_masks.unsqueeze(1)
        x = x.masked_fill(mask, 0.0)

        if contexts.size(2) > 0:
            x = torch.cat([contexts, x], dim=2)
            cache = x[:, :, -self.left_context :]
        else:
            x = F.pad(x, (self.left_context, 0))
            cache = contexts

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation2(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)

        return x, cache


class FeedForwardModule(nn.Module):
    r"""Feed forward module.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float): dropout probability.
        epsilon (float, optional): LayerNorm epsilon. (Default: 1e-5)
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, dropout: float, epsilon: float = 1e-5
    ) -> None:
        super().__init__()

        self.sequential = nn.Sequential(
            nn.LayerNorm(input_dim, eps=epsilon),
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
        return self.sequential(inputs)


class ConformerLayer(nn.Module):
    r"""The layer that constitutes Conformer model.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_heads (int): number of attention heads.
        kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float): dropout probability.
        use_cross_attn (bool, optional): use cross-attention module. (Default: False)
        cross_attn_dim (int, optional): dimension of cross-attention keys/values. (Default: 256)
        epsilon (float, optional): LayerNorm epsilon. (Default: 1e-5)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_heads: int,
        kernel_size: int,
        dropout: float,
        use_cross_attn: bool = False,
        cross_attn_dim: int = 256,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()

        self.ffn1_module = FeedForwardModule(input_dim, ffn_dim, dropout, epsilon)
        self.attn_module = SelfAttentionModule(input_dim, num_heads, dropout, epsilon)
        self.conv_module = ConvolutionModule(input_dim, kernel_size, dropout, epsilon)
        self.ffn2_module = FeedForwardModule(input_dim, ffn_dim, dropout, epsilon)
        self.layer_norm = nn.LayerNorm(input_dim, eps=epsilon)

        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            self.cross_attn_module = CrossAttentionModule(
                input_dim, num_heads, cross_attn_dim, dropout, epsilon
            )

    def forward(
        self,
        x: torch.Tensor,
        conv_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        conv_cache: torch.Tensor,
        attn_cache: torch.Tensor,
        prompt: torch.Tensor = None,
        prompt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Args:
            x (Tensor): input, with shape `(B, T, D)`.
            conv_mask (Tensor): convolution mask, with shape `(B, T)`.
            attn_mask (Tensor): attention mask, with shape `(T, T + left_context)`.
            conv_cache (Tensor): convolution cache, with shape `(B, D, kernel_size - 1)`.
            attn_cache (Tensor): attention cache, with shape `(B, left_context, D)`.
            prompt (torch.Tensor, optional): prompt features, with shape `(B, T', D')`.
            prompt_mask (torch.Tensor, optional): padding mask for prompts, with shape `(B, T')`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """

        residual = x
        x = self.ffn1_module(x)
        x = x * 0.5 + residual

        residual = x
        x, attn_cache = self.attn_module(x, conv_mask, attn_mask, attn_cache)
        x = x + residual

        if self.use_cross_attn and prompt is not None:
            x = x + self.cross_attn_module(x, prompt, conv_mask, prompt_mask)

        residual = x
        x, conv_cache = self.conv_module(x, conv_mask, conv_cache)
        x = x + residual

        residual = x
        x = self.ffn2_module(x)
        x = x * 0.5 + residual

        x = self.layer_norm(x)

        return x, conv_cache, attn_cache
