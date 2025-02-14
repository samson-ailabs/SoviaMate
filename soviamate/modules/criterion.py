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
