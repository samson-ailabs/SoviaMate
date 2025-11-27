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

"""Feature extractions for speech processing pipelines"""

from typing import List
import torch
from torchaudio import transforms as T


class LogMelFilterbank:
    r"""Extract log mel filterbank features from the waveform.

    Args:
        sample_rates (int | List[int]): sample rates of the input waveforms
        n_fft (int | float): size of the FFT
        win_length (int | float): size of the window
        hop_length (int | float): size of the hop
        n_mels (int): number of mel bands
    """

    def __init__(
        self,
        sample_rates: int | List[int],
        n_fft: float,
        win_length: float,
        hop_length: float,
        n_mels: int,
    ):
        if isinstance(sample_rates, int):
            sample_rates = [sample_rates]

        self.extractors = {
            sample_rate: T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=int(n_fft * sample_rate),
                win_length=int(win_length * sample_rate),
                hop_length=int(hop_length * sample_rate),
                n_mels=n_mels,
                center=False,
            )
            for sample_rate in sample_rates
        }

    def extract(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        r"""Extract log mel filterbank features from the waveform.

        Args:
            waveform (Tensor): input waveform, shape (1, L)
            sample_rate (int): sample rate of the waveform

        Returns:
            Tensor: log mel filterbank features, shape (n_mels, T)
        """

        if sample_rate not in self.extractors:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")

        fbank = self.extractors[sample_rate](waveform)
        fbank = fbank.squeeze(0).clamp(1e-5).log()

        return fbank
