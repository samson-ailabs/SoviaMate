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

"""Loss Functions for different tasks"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from soviamate.utils.helper import make_padding_mask


class LeastSquaresAdversarialLoss(nn.Module):
    r"""Least Squares GAN loss for multi-scale discriminators."""

    def forward(
        self,
        fake_logits: List[torch.Tensor],
        real_logits: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        r"""Compute the least squares GAN loss.

        Args:
            fake_logits: List of discriminator outputs for fake samples.
            real_logits: List of discriminator outputs for real samples.
                If None, computes generator loss; otherwise discriminator loss.

        Returns:
            Tensor: Least squares GAN loss.
        """
        loss = 0.0

        if real_logits is None:
            # Generator loss: fake should be classified as real
            for fake in fake_logits:
                loss += torch.mean((fake - 1) ** 2)
        else:
            # Discriminator loss: real -> 1, fake -> 0
            for real, fake in zip(real_logits, fake_logits):
                loss += torch.mean((real - 1) ** 2) + torch.mean(fake**2)

        return loss / len(fake_logits)


class MelSpectralEnergyDistanceLoss(nn.Module):
    """Multi-scale mel spectral energy distance loss from SpectroStream.

    Args:
        sample_rate: Audio sample rate in Hz.
        window_sizes: List of STFT window sizes.
        mel_bins: List of mel bins corresponding to each window size.
    """

    def __init__(self, sample_rate: int, window_sizes: List[int], mel_bins: List[int]):
        super().__init__()

        self.register_buffer(
            "alphas", torch.tensor([math.sqrt(s / 2) for s in window_sizes])
        )

        self.mel_spectrograms = nn.ModuleList()
        for window_size, n_mels in zip(window_sizes, mel_bins):
            self.mel_spectrograms.append(
                T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=window_size,
                    win_length=window_size,
                    hop_length=window_size // 4,
                    n_mels=n_mels,
                    power=1.0,
                )
            )

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute mel spectral energy distance loss.

        Args:
            outputs: Predicted waveforms of shape `(B, T)` or `(B, 1, T)`.
            targets: Target waveforms of shape `(B, T)` or `(B, 1, T)`.
            lengths: Valid lengths per sample of shape `(B,)`.

        Returns:
            Scalar loss value summed across all scales.
        """
        # Handle channel dimension
        if outputs.dim() == 3:
            outputs = outputs.squeeze(1)
        if targets.dim() == 3:
            targets = targets.squeeze(1)

        # Create mask from waveform lengths
        masks = make_padding_mask(lengths)
        masks = (~masks[:, None, :]).float()

        l1_loss = 0.0
        l2_loss = 0.0

        for i, mel_spec in enumerate(self.mel_spectrograms):
            # Compute mel spectrograms: (B, n_mels, T_mel)
            mel_output = mel_spec(outputs)
            mel_target = mel_spec(targets)

            # Resize mask to match spectrogram dimensions
            mask = F.interpolate(masks, size=mel_output.size(2))
            mask = mask.expand_as(mel_output)

            # Number of valid elements
            numels = mask.sum().clamp(min=1.0)

            l1 = F.l1_loss(mel_output, mel_target, reduction="none")
            l1_loss += (l1 * mask).sum() / numels

            log_mel_output = mel_output.clamp(min=1e-5).log()
            log_mel_target = mel_target.clamp(min=1e-5).log()

            l2 = F.mse_loss(log_mel_output, log_mel_target, reduction="none")
            l2_loss += self.alphas[i] * torch.sqrt((l2 * mask).sum()) / numels

        return l1_loss + l2_loss


class SequenceToSequenceLoss(nn.Module):
    r"""Loss function for sequence-to-sequence models.

    Args:
        blank (int, optional): Index of the blank token. Default: 0.
        zero_infinity (bool, optional): Whether to zero infinite losses. Default: True.
    """

    def __init__(self, blank: int = 0, zero_infinity: bool = True) -> None:
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)

    def forward(
        self,
        logits: torch.Tensor,
        logit_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the sequence-to-sequence loss using CTC loss.

        Args:
            logits (Tensor): Decoder logits with shape `(B, T, vocab_size)`.
            logit_lengths (Tensor): Lengths of decoder outputs with shape `(B,)`.
            targets (Tensor): Target token sequences with shape `(B, U)`.
            target_lengths (Tensor): Lengths of target sequences with shape `(B,)`.

        Returns:
            Tensor: Loss value (scalar).
        """
        log_probs = F.log_softmax(logits.permute(1, 0, 2), dim=2)
        loss = self.ctc_loss(log_probs, targets, logit_lengths, target_lengths)

        return loss


class SequenceContrastiveLoss(nn.Module):
    """VICReg-based contrastive loss for learning invariant sequence representations.

    Combines three objectives to learn robust representations without negative samples:
    - Invariance: Align representations of paired sequences
    - Variance: Maintain per-dimension variance to prevent collapse
    - Covariance: Decorrelate dimensions for diverse features

    Args:
        input_dim (int): Input feature dimension.
        projector_dim (int): Projector MLP hidden and output dimension.
        lambda_inv (float): Invariance loss weight.
        lambda_var (float): Variance loss weight.
        lambda_cov (float): Covariance loss weight.
        variance_gamma (float): Minimum std threshold for variance hinge loss.
        variance_epsilon (float): Numerical stability constant for variance.
    """

    def __init__(
        self,
        input_dim: int,
        projector_dim: int = 4096,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        variance_gamma: float = 1.0,
        variance_epsilon: float = 1e-4,
    ):
        super().__init__()

        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.variance_gamma = variance_gamma
        self.variance_epsilon = variance_epsilon

        self.projector = nn.Sequential(
            nn.Conv1d(input_dim, projector_dim, kernel_size=1),
            nn.InstanceNorm1d(projector_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(projector_dim, projector_dim, kernel_size=1),
            nn.InstanceNorm1d(projector_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(projector_dim, projector_dim, kernel_size=1),
        )

    def invariance_loss(
        self, z_a: torch.Tensor, z_b: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute invariance loss between paired sequences.

        Args:
            z_a (Tensor): First sequence embeddings of shape (B, T, D).
            z_b (Tensor): Second sequence embeddings of shape (B, T, D).
            mask (Tensor): Valid frame mask of shape (B, T).

        Returns:
            Tensor: Masked MSE loss averaged across batch.
        """
        diff = F.mse_loss(z_a, z_b, reduction="none")
        masked_diff = diff * mask.unsqueeze(-1)

        num_valid = mask.unsqueeze(-1).expand_as(diff).sum(dim=(1, 2))
        inv_loss = (masked_diff.sum(dim=(1, 2)) / num_valid).mean()

        return inv_loss

    def variance_covariance_loss(
        self, z: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute variance and covariance losses per utterance.

        Args:
            z (Tensor): Embeddings of shape (B, T, D).
            mask (Tensor): Valid frame mask of shape (B, T).
            lengths (Tensor): Valid frame counts of shape (B,).

        Returns:
            Tensor: Variance loss (scalar).
            Tensor: Covariance loss (scalar).
        """
        mask = mask.unsqueeze(-1)
        lengths = lengths.unsqueeze(1) - 1

        # Center per utterance
        mean = (z * mask).sum(dim=1) / lengths
        centered = z - mean.unsqueeze(1)
        masked_centered = centered * mask

        # Variance: hinge loss on std per dimension
        var = (centered.pow(2) * mask).sum(dim=1) / (lengths - 1).clamp(min=1.0)
        std = torch.sqrt(var + self.variance_epsilon)
        loss_var = torch.relu(self.variance_gamma - std).mean()

        # Covariance: penalize off-diagonal correlations
        cov = torch.einsum("btd,bte->bde", masked_centered, masked_centered)
        cov = cov / (lengths.view(-1, 1, 1) - 1).clamp(min=1.0)

        diag_sum = cov.diagonal(dim1=1, dim2=2).pow(2).sum(dim=1)
        total_sum = cov.pow(2).sum(dim=(1, 2))
        loss_cov = ((total_sum - diag_sum) / z.size(2)).mean()

        return loss_var, loss_cov

    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VICReg loss for paired sequences.

        Args:
            source_embeddings (Tensor): First sequence of shape (B, T, D).
            target_embeddings (Tensor): Second sequence of shape (B, T, D).
            source_lengths (Tensor): Valid frame counts for source of shape (B,).
            target_lengths (Tensor): Valid frame counts for target of shape (B,).

        Returns:
            Tensor: Weighted sum of invariance, variance, and covariance losses.
        """
        assert source_lengths.equal(target_lengths), "Lengths must match"
        batch_size = source_embeddings.size(0)

        # Project both views through shared MLP
        stacked = torch.cat([source_embeddings, target_embeddings], dim=0)
        proj = self.projector(stacked.transpose(1, 2)).transpose(1, 2)

        # Split back into source and target
        src_proj, tgt_proj = proj[:batch_size], proj[batch_size:]

        # Compute mask from lengths
        mask = ~make_padding_mask(source_lengths)

        # Invariance: pull corresponding frames together
        loss_inv = self.invariance_loss(src_proj, tgt_proj, mask)

        # Variance & covariance: average over both views
        var_src, cov_src = self.variance_covariance_loss(src_proj, mask, source_lengths)
        var_tgt, cov_tgt = self.variance_covariance_loss(tgt_proj, mask, target_lengths)

        loss_var = (var_src + var_tgt) / 2
        loss_cov = (cov_src + cov_tgt) / 2

        return (
            self.lambda_inv * loss_inv
            + self.lambda_var * loss_var
            + self.lambda_cov * loss_cov
        )
