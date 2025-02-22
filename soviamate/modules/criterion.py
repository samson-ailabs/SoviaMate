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

from soviamate.layers.recognizer import Predictor, Joint
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


class MultiResolutionSTFTLoss(nn.Module):
    r"""Multi-resolution STFT loss.

    Args:
        n_ffts (List[int]): list of the number of Fourier bins.
        win_lengths (List[int]): list of the window lengths.
        hop_lengths (List[int]): list of the hop lengths.
    """

    def __init__(
        self, n_ffts: List[int], win_lengths: List[int], hop_lengths: List[int]
    ) -> None:
        super().__init__()

        self.spectrograms = nn.ModuleList()
        for n_fft, win_length, hop_length in zip(n_ffts, win_lengths, hop_lengths):
            self.spectrograms.append(
                T.Spectrogram(n_fft, win_length, hop_length, power=1)
            )

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute the multi-resolution STFT loss.

        Args:
            outputs (Tensor): waveform outputs, shape (B, 1, T).
            targets (Tensor): waveform targets, shape (B, 1, T).
            lengths (Tensor): lengths of the waveform targets, shape (B,).

        Returns:
            Tensor: multi-resolution STFT loss.
        """

        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)

        masks = make_padding_mask(lengths)
        masks = ~masks[:, None, :]

        loss = 0.0
        for spectrogram in self.spectrograms:
            loss += self._forward_spectrogram(
                spectrogram(outputs), spectrogram(targets), masks
            )

        loss = loss / len(self.spectrograms)

        return loss

    def _forward_spectrogram(
        self, spec_outs: torch.Tensor, spec_tgts: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:

        masks = F.interpolate(masks.float(), spec_tgts.size(2))
        masks = masks.bool().expand_as(spec_tgts)

        sc_loss = self._spectral_convergence_loss(spec_outs, spec_tgts, masks)
        mag_loss = self._log_stft_magnitude_loss(spec_outs, spec_tgts, masks)

        loss = sc_loss + mag_loss

        return loss

    def _spectral_convergence_loss(
        self, xs: torch.Tensor, ys: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:

        numerator = ((ys - xs) * masks).norm(p="fro")
        denominator = (ys * masks).norm(p="fro")

        loss = numerator / (denominator + 1e-9)

        return loss

    def _log_stft_magnitude_loss(
        self, xs: torch.Tensor, ys: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:

        xs = xs.add(1e-9).log()
        ys = ys.add(1e-9).log()

        loss = F.l1_loss(xs, ys, reduction="none")
        loss = (loss * masks).sum() / masks.sum()

        return loss


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
