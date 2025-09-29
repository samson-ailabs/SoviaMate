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

from soviamate.layers.recognizer import Joint, Predictor
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


class HingeGANLoss(nn.Module):
    """Hinge loss for Generative Adversarial Networks.

    Implements the classical GAN objective with hinge loss for discriminator and generator
    as described in "Spectral Normalization for Generative Adversarial Networks" and other papers.

    For discriminator: L_D(x) = (1/N) * sum_n [max(0, 1 - D_n(x)) + max(0, 1 + D_n(G(x)))]
    For generator: L_G^adv(x) = (1/N) * sum_n max(0, 1 - D_n(G(x)))

    Where:
    - N is the number of logits (combined from multiple discriminators if applicable)
    - D_n(x) is the n-th logit output
    """

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute hinge loss for discriminator or generator.

        Args:
            outputs (Tensor): discriminator outputs for fake samples.
            targets (Tensor, optional): discriminator outputs for real samples.
                If None, computes generator loss. If provided, computes discriminator loss.

        Returns:
            Tensor: Hinge loss value.
        """

        if targets is None:
            # Generator loss: max(0, 1 - D(fake))
            return torch.mean(F.relu(1.0 - outputs))

        else:
            # Discriminator loss: max(0, 1 - D(real)) + max(0, 1 + D(fake))
            return torch.mean(F.relu(1.0 - targets)) + torch.mean(F.relu(1.0 + outputs))


class FeatureMatchingLoss(nn.Module):
    """Feature Matching Loss for GANs following SpectroStream paper.

    Implements equation (5) from SpectroStream:
    L_feat_G(x) = 1/(KL) * Σ(k,l) 1/M_k,l * ||D_k^(l)(x) - D_k^(l)(G(x))||_1

    Where K is number of discriminator scales, L is number of intermediate layers,
    and M_k,l is the size of the l-th intermediate output tensor.
    """

    def forward(
        self, fake_features: List[torch.Tensor], real_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            fake_features (List[Tensor]): List of features from discriminator for fake samples
            real_features (List[Tensor]): List of features from discriminator for real samples

        Returns:
            Tensor: Feature matching loss following SpectroStream formulation.
        """

        total_loss = 0.0

        for fake_feat, real_feat in zip(fake_features, real_features):
            # L1 loss normalized by tensor size: 1/M_k,l * ||D_k^(l)(x) - D_k^(l)(G(x))||_1
            loss = F.l1_loss(fake_feat, real_feat, reduction="none")
            total_loss += loss.sum() / fake_feat.numel()

        # SpectroStream averages over all feature maps: 1/(K*L) normalization
        # where K*L = total number of feature maps from all discriminator scales and layers
        num_feature_maps = len(fake_features)
        total_loss = total_loss / num_feature_maps

        return total_loss


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
            outputs (Tensor): waveform outputs, shape (B, 1, T)
            targets (Tensor): waveform targets, shape (B, 1, T)
            lengths (Tensor): lengths of the waveform targets, shape (B,)

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
    r"""Integrate CTC and RNN-T loss for training sequence-to-sequence models.

    Args:
        embedding_dim (int): dimension of the embedding layer.
        encoder_dim (int): dimension of the encoder outputs.
        predictor_dim (int): dimension of the predictor outputs.
        joint_dim (int): dimension of the joint outputs.
        context_size (int): size of the context window.
        vocab_size (int): size of the vocabulary.
        dropout (float): dropout probability.
        num_samples (int): number of samples for sampled softmax.
    """

    def __init__(
        self,
        embedding_dim: int,
        encoder_dim: int,
        predictor_dim: int,
        joint_dim: int,
        context_size: int,
        vocab_size: int,
        dropout: float,
        num_samples: int,
    ) -> None:
        super().__init__()
        self.blank_token = 0

        self.vocab_size = vocab_size
        self.num_samples = num_samples

        self.predictor = Predictor(
            embedding_dim, predictor_dim, vocab_size, context_size, dropout
        )

        self.joint = Joint(encoder_dim, predictor_dim, joint_dim)
        self.linear = nn.Linear(joint_dim, vocab_size)

        self.ctc_loss = nn.CTCLoss(blank=self.blank_token, zero_infinity=True)
        self.rnnt_loss = T.RNNTLoss(blank=self.blank_token, fused_log_softmax=True)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: torch.Tensor,
        decoder_outputs: torch.Tensor,
        decoder_lengths: torch.Tensor,
        target_tokens: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the sequence-to-sequence loss.

        Args:
            encoder_outputs (Tensor): encoder outputs, shape `(B, T, D)`.
            encoder_lengths (Tensor): lengths of the encoder outputs, shape `(B,)`.
            decoder_outputs (Tensor): decoder outputs, shape `(B, T, D)`.
            decoder_lengths (Tensor): lengths of the decoder outputs, shape `(B,)`.
            target_tokens (Tensor): target sequences, shape `(B, U)`.
            target_lengths (Tensor): lengths of the target sequences, shape `(B,)`.

        Returns:
            Tensor: combined loss of CTC and RNN-T.
        """

        log_probs = decoder_outputs.permute(1, 0, 2)
        log_probs = F.log_softmax(log_probs, dim=2)

        ctc_loss = self.ctc_loss(
            log_probs, target_tokens, decoder_lengths, target_lengths
        )

        predictor_outputs, _ = self.predictor(target_tokens, target_lengths)
        joint_outputs = self.joint(encoder_outputs, predictor_outputs)

        with torch.no_grad():
            pad_masks = make_padding_mask(decoder_lengths)
            pad_masks = (~pad_masks)[:, :, None]

            ctc_probs = (decoder_outputs * pad_masks).sum(1)
            ctc_probs = ctc_probs / decoder_lengths.unsqueeze(1)

            batch_idxs = torch.arange(ctc_probs.size(0), device=ctc_probs.device)
            ctc_probs[batch_idxs.unsqueeze(1), target_tokens] = float("inf")

            top_tokens = ctc_probs[:, 1:].topk(self.num_samples, dim=1)[1]
            top_tokens = top_tokens.sort(dim=1).values + 1

            samples = F.pad(top_tokens, (1, 0), value=self.blank_token)

        targets = target_tokens[:, :, None] > samples[:, None, :]
        targets = targets.sum(dim=2).clamp(min=1).type(torch.int32)

        weight = self.linear.weight[samples]
        bias = self.linear.bias[samples]

        logits = torch.einsum("btuv,bsv->btus", joint_outputs, weight)
        logits = logits + bias[:, None, None, :]

        rnnt_loss = self.rnnt_loss(
            logits, targets, encoder_lengths.int(), target_lengths.int()
        )

        loss = ctc_loss + rnnt_loss

        return loss
