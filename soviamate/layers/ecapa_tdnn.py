# Copyright (c) 2025, Son Dang Dinh. All rights reserved.
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#
# Vendored from UniSpeech and refactored to accept pre-extracted hidden
# states from HuggingFace WavLM, removing the s3prl/fairseq dependency
# while maintaining weight compatibility with wavlm_large_finetune.pth.
#
# Original: https://github.com/microsoft/UniSpeech/blob/main/downstreams/
#           speaker_verification/models/ecapa_tdnn.py
#
# IMPORTANT: nn.Module attribute names (e.g. self.Conv1dReluBn1,
# self.SE_Connect) MUST match the original checkpoint keys exactly.

"""ECAPA-TDNN speaker verification head (vendored from UniSpeech)."""

import torch
from torch import nn
from torch.nn import functional as F


class Res2Conv1dReluBn(nn.Module):
    """Multi-scale residual 1D convolution (Res2Net style)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        scale: int = 4,
    ):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList(
            nn.Conv1d(
                self.width,
                self.width,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=bias,
            )
            for _ in range(self.nums)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.width) for _ in range(self.nums))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale residual convolution."""
        splits = torch.split(x, self.width, dim=1)
        out = []
        residual = splits[0]
        for i in range(self.nums):
            residual = splits[i] if i == 0 else residual + splits[i]
            residual = self.bns[i](F.relu(self.convs[i](residual)))
            out.append(residual)
        if self.scale != 1:
            out.append(splits[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    """Conv1d followed by ReLU and BatchNorm1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv, relu, batchnorm."""
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):  # noqa: N801  # pylint: disable=invalid-name
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, se_bottleneck_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-and-excitation gating."""
        s = torch.sigmoid(self.linear2(F.relu(self.linear1(x.mean(dim=2)))))
        return x * s.unsqueeze(2)


class SE_Res2Block(nn.Module):  # noqa: N801  # pylint: disable=invalid-name
    """SE-Res2Block of the ECAPA-TDNN architecture."""

    # Attribute names MUST match checkpoint keys — do not rename.
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        scale: int,
        se_bottleneck_dim: int,
    ):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(  # noqa: N815  # pylint: disable=invalid-name
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(  # noqa: N815  # pylint: disable=invalid-name
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            scale=scale,
        )
        self.Conv1dReluBn2 = Conv1dReluBn(  # noqa: N815  # pylint: disable=invalid-name
            out_channels,
            out_channels,
            kernel_size=1,
        )
        self.SE_Connect = SE_Connect(  # noqa: N815  # pylint: disable=invalid-name
            out_channels,
            se_bottleneck_dim,
        )
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE-Res2 block with residual connection."""
        residual = self.shortcut(x) if self.shortcut else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    """Attentive weighted mean and standard deviation pooling."""

    def __init__(self, in_dim: int, attention_channels: int = 128):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attentive weighted mean and std pooling."""
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x))), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * x**2, dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPATDNNHead(nn.Module):
    """ECAPA-TDNN speaker embedding head for WavLM hidden states.

    Weight-compatible with ``wavlm_large_finetune.pth`` from UniSpeech.
    Attribute names match the original ``ECAPA_TDNN`` class so
    that ``load_state_dict`` works directly with the checkpoint (after
    stripping the ``feature_extract.*`` prefix).

    Args:
        feat_dim: Hidden state dimension from WavLM (1024 for WavLM-Large).
        channels: Internal channel width (512 for ECAPA-TDNN-SMALL).
        emb_dim: Output embedding dimension (256).
        num_feat_layers: Number of WavLM hidden state layers (25 for Large).
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        channels: int = 512,
        emb_dim: int = 256,
        num_feat_layers: int = 25,
    ):
        super().__init__()

        # Learned weighted sum over WavLM hidden state layers
        self.feature_weight = nn.Parameter(torch.zeros(num_feat_layers))
        self.instance_norm = nn.InstanceNorm1d(feat_dim)

        # ECAPA-TDNN layers
        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            channels,
            channels,
            3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer3 = SE_Res2Block(
            channels,
            channels,
            3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer4 = SE_Res2Block(
            channels,
            channels,
            3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
            se_bottleneck_dim=128,
        )

        # Aggregation: concat layers 2-4, pool, project
        self.conv = nn.Conv1d(channels * 3, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, attention_channels=128)
        self.bn = nn.BatchNorm1d(3072)
        self.linear = nn.Linear(3072, emb_dim)

    def forward(
        self,
        hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Extract speaker embedding from WavLM hidden states.

        Args:
            hidden_states: Sequence of *num_feat_layers* tensors, each
                ``(B, T, feat_dim)``.  Obtained via
                ``WavLMModel(..., output_hidden_states=True).hidden_states``.

        Returns:
            Speaker embedding of shape ``(B, emb_dim)``.
        """
        # Learned weighted sum across layers
        x = torch.stack(list(hidden_states), dim=0)  # (L, B, T, D)
        w = F.softmax(self.feature_weight, dim=-1).view(-1, 1, 1, 1)
        x = (w * x).sum(dim=0)  # (B, T, D)
        x = self.instance_norm(x.transpose(1, 2) + 1e-6)  # (B, D, T)

        # ECAPA-TDNN with multi-layer aggregation
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = F.relu(self.conv(torch.cat([out2, out3, out4], dim=1)))
        out = self.bn(self.pooling(out))

        return self.linear(out)
