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

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from soviamate.utils.helper import make_padding_mask


class LeastSquaresGANLoss(nn.Module):
    r"""Least Squares GAN loss for Generative Adversarial Networks."""

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Compute the least squares GAN loss.

        Args:
            outputs (Tensor): outputs of the discriminator for fake samples.
            targets (Tensor, optional): outputs of the discriminator for real samples.

        Returns:
            Tensor: Least squares GAN loss.
        """

        if targets is None:
            loss = torch.mean((outputs - 1) ** 2)
        else:
            loss = torch.mean((targets - 1) ** 2) + torch.mean(outputs**2)

        return loss


class MelSpectralEnergyLoss(nn.Module):
    """
    Multi-Resolution Mel-Spectrogram Loss for audio reconstruction.

    Combines both mel-spectrogram reconstruction loss and multi-scale spectral losses
    as described in the research. Uses variable window sizes with proportional hop lengths
    and mel bin sizes to better capture frequency information at multiple time-scales.

    Args:
        sample_rate (int): Sample rate of the audio signals.
        window_lengths (List[int]): List of window lengths for STFT.
        mel_bins (List[int]): List of mel bin sizes corresponding to each window length.
    """

    def __init__(self, sample_rate: int, fft_sizes: List[int], mel_bins: List[int]):
        super().__init__()

        assert len(fft_sizes) == len(mel_bins), (
            "fft_sizes and mel_bins must have same length"
        )

        self.sample_rate = sample_rate
        self.fft_sizes = fft_sizes
        self.mel_bins = mel_bins

        self.mel_spectrograms = nn.ModuleList()
        for fft_size, n_mels in zip(fft_sizes, mel_bins):
            self.mel_spectrograms.append(
                T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=fft_size,
                    win_length=fft_size,
                    hop_length=fft_size // 4,
                    n_mels=n_mels,
                    power=1.0,
                    norm="slaney",
                    mel_scale="slaney",
                )
            )

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the multi-resolution mel-spectrogram loss.

        Args:
            outputs (Tensor): waveform outputs, shape `(B, 1, T)`.
            targets (Tensor): waveform targets, shape `(B, 1, T)`.
            lengths (Tensor): lengths of the waveform targets, shape `(B,)`.

        Returns:
            Tensor: Combined multi-resolution mel-spectrogram loss
        """

        # Remove channel dimension if present
        if outputs.dim() == 3:
            outputs = outputs.squeeze(1)
        if targets.dim() == 3:
            targets = targets.squeeze(1)

        # Create masks from lengths if provided
        masks = make_padding_mask(lengths)
        masks = (~masks[:, None, :]).float()

        # Calculate loss at each resolution
        total_loss = 0.0

        for mel_spectrogram in self.mel_spectrograms:
            # Compute mel spectrograms
            mel_outputs = mel_spectrogram(outputs).clamp(min=1e-5).log()
            mel_targets = mel_spectrogram(targets).clamp(min=1e-5).log()

            # Resize mask to match spectrogram dimensions
            curr_masks = F.interpolate(masks, size=mel_outputs.size(2))
            curr_masks = curr_masks.bool().expand_as(mel_outputs)

            # Compute L1 loss between mel spectrograms
            loss = F.l1_loss(mel_outputs, mel_targets, reduction="none")
            loss = (loss * curr_masks).sum() / curr_masks.sum()

            # Combine losses across resolutions
            total_loss += loss

        return total_loss


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


class InformationContrastiveLoss(nn.Module):
    """Information Noise-Contrastive Estimation loss for representation learning.

    Implements the InfoNCE objective for learning robust representations via
    contrastive learning. Maximizes agreement between positive pairs while
    pushing apart negative samples in the batch.

    Args:
        embed_dim (int): Input embedding dimension.
        temperature (float): Temperature parameter for scaling logits.
        projection_dim (int): Projection head output dimension. If None, same as embed_dim.
        projection_hidden_dim (int): Projection head hidden dimension. If None, 2x embed_dim.
    """

    def __init__(
        self,
        embed_dim: int,
        temperature: float = 0.1,
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.temperature = temperature

        proj_dim = projection_dim or embed_dim
        hidden_dim = projection_hidden_dim or embed_dim * 2

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        source_lengths: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute contrastive loss for paired embeddings.

        Args:
            source_embeddings (Tensor): First set of embeddings of shape `(B, T, D)`.
            target_embeddings (Tensor): Second set of embeddings of shape `(B, T, D)`.
            source_lengths (Tensor, optional): Valid lengths of source `(B,)`.
            target_lengths (Tensor, optional): Valid lengths of target `(B,)`.

        Returns:
            Tensor: Contrastive loss value (scalar).
        """
        # Pool source embeddings over time dimension
        if source_lengths is not None:
            masks = make_padding_mask(source_lengths)
            masks = (~masks).float().unsqueeze(-1)
            source_pooled = (source_embeddings * masks).sum(dim=1) / masks.sum(dim=1)
        else:
            source_pooled = source_embeddings.mean(dim=1)

        # Pool target embeddings over time dimension
        if target_lengths is not None:
            masks = make_padding_mask(target_lengths)
            masks = (~masks).float().unsqueeze(-1)
            target_pooled = (target_embeddings * masks).sum(dim=1) / masks.sum(dim=1)
        else:
            target_pooled = target_embeddings.mean(dim=1)

        # Apply projection head after pooling
        source_pooled = self.projector(source_pooled)
        target_pooled = self.projector(target_pooled)

        # Normalize embeddings to unit sphere
        source_pooled = F.normalize(source_pooled, p=2, dim=1)
        target_pooled = F.normalize(target_pooled, p=2, dim=1)

        # Compute cosine similarity (B x B)
        similarity = torch.mm(source_pooled, target_pooled.T) / self.temperature

        # Labels: src[i] matches tgt[i]
        labels = torch.arange(source_pooled.size(0), device=source_pooled.device)

        # Compute loss in both directions and average
        loss_s2t = F.cross_entropy(similarity, labels)
        loss_t2s = F.cross_entropy(similarity.T, labels)

        return (loss_s2t + loss_t2s) / 2


class SequenceContrastiveLoss(nn.Module):
    """Contrastive loss for learning invariant sequence representations.

    Implements VICReg (Bardes et al., 2022) adapted for sequential data.
    Unlike InfoNCE, VICReg avoids representation collapse without requiring
    negative samples by combining three complementary objectives:

    - **Invariance**: Aligns representations of paired source/target frames
    - **Variance**: Prevents collapse by maintaining per-dimension variance
    - **Covariance**: Encourages diverse features by decorrelating dimensions

    Samples frames randomly across batch for efficient statistics computation,
    using the same indices for source and target to preserve alignment.

    Args:
        num_samples (int): Number of frames to sample across batch for statistics.
        lambda_inv (float): Weight for invariance term (alignment strength).
        lambda_var (float): Weight for variance term (prevents collapse).
        lambda_cov (float): Weight for covariance term (feature decorrelation).
        variance_gamma (float): Minimum variance threshold for hinge loss.
        variance_epsilon (float): Numerical stability for variance computation.
    """

    def __init__(
        self,
        num_samples: int = 1024,
        lambda_inv: float = 5.0,
        lambda_var: float = 1.0,
        lambda_cov: float = 1.0,
        variance_gamma: float = 1.0,
        variance_epsilon: float = 1e-4,
    ):
        super().__init__()

        self.num_samples = num_samples
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.variance_gamma = variance_gamma
        self.variance_epsilon = variance_epsilon

    def invariance_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute invariance loss: MSE between aligned frame pairs.

        Args:
            z_a: First view embeddings of shape `(N, D)`.
            z_b: Second view embeddings of shape `(N, D)`.

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(z_a, z_b, reduction="mean")

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute variance loss: hinge loss on per-dimension standard deviation.

        Prevents representation collapse by enforcing a minimum variance threshold
        for each dimension. Only penalizes dimensions with std below gamma.

        Args:
            z: Embeddings of shape `(N, D)`.

        Returns:
            Scalar variance loss.
        """
        # Compute std per dimension (across num_samples)
        std = torch.sqrt(z.var(dim=0) + self.variance_epsilon)

        # Hinge loss: penalize only if std < gamma
        loss = torch.relu(self.variance_gamma - std).mean()

        return loss

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute covariance loss: penalize off-diagonal covariance elements.

        Encourages decorrelated features by minimizing correlations between
        different dimensions, promoting diverse and non-redundant representations.

        Args:
            z: Embeddings of shape `(N, D)`.

        Returns:
            Scalar covariance loss.
        """
        num_samples, dim = z.shape

        # Center embeddings
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Covariance matrix: C = (1/(n-1)) * Z^T @ Z
        cov = (z_centered.T @ z_centered) / (num_samples - 1)

        # Penalize off-diagonal elements only
        off_diagonal = cov.pow(2).sum() - cov.diag().pow(2).sum()
        loss = off_diagonal / dim

        return loss

    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | dict:
        """Compute contrastive loss with cross-batch frame sampling.

        Samples frames randomly across the batch while using the same indices
        for both source and target to preserve temporal alignment.

        Args:
            source_embeddings (Tensor): Source frames of shape `(B, T, D)`.
            target_embeddings (Tensor): Target frames of shape `(B, T, D)`.
            source_lengths (Tensor): Valid frame counts of shape `(B,)`.
            target_lengths (Tensor): Valid frame counts of shape `(B,)`.
            return_components (bool): If True, return dict with loss components.

        Returns:
            Scalar loss or dict with individual loss components for logging.
        """
        assert source_lengths.equal(target_lengths), (
            "Source and target lengths must be equal for aligned frames."
        )

        embed_dim = source_embeddings.shape[-1]
        device = source_embeddings.device

        # Create mask for valid frames (source and target have same length)
        mask = make_padding_mask(source_lengths)
        valid_mask = (~mask).reshape(-1)  # (B x T)

        # Flatten to (B*T, D) and extract valid frames
        src_flat = source_embeddings.reshape(-1, embed_dim)
        tgt_flat = target_embeddings.reshape(-1, embed_dim)

        src_valid = src_flat[valid_mask]
        tgt_valid = tgt_flat[valid_mask]

        # Random sampling (same indices for source and target alignment)
        num_valid = src_valid.shape[0]

        if num_valid <= self.num_samples:
            src_sampled = src_valid
            tgt_sampled = tgt_valid
        else:
            # Sample same random indices from both
            indices = torch.randperm(num_valid, device=device)
            indices = indices[: self.num_samples]

            src_sampled = src_valid[indices]
            tgt_sampled = tgt_valid[indices]

        # Compute loss components
        loss_inv = self.invariance_loss(src_sampled, tgt_sampled)

        # Variance and covariance computed on both views and averaged
        loss_var_src = self.variance_loss(src_sampled)
        loss_var_tgt = self.variance_loss(tgt_sampled)
        loss_var = (loss_var_src + loss_var_tgt) / 2

        loss_cov_src = self.covariance_loss(src_sampled)
        loss_cov_tgt = self.covariance_loss(tgt_sampled)
        loss_cov = (loss_cov_src + loss_cov_tgt) / 2

        # Combined weighted loss
        total_loss = (
            self.lambda_inv * loss_inv
            + self.lambda_var * loss_var
            + self.lambda_cov * loss_cov
        )

        if return_components:
            return {
                "total_loss": total_loss,
                "invariance": loss_inv,
                "variance": loss_var,
                "covariance": loss_cov,
            }

        return total_loss
