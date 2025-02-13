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

"""Audio and feature transformations for data augmentation"""

import random

import torch
import torchaudio


class TimeStretch:
    r"""Change the speed of the audio without modifying the pitch.

    Args:
        min_speed_rate: Minimum speed rate to change the audio.
        max_speed_rate: Maximum speed rate to change the audio.
        probability: Probability of applying the speed perturbation
    """

    def __init__(
        self, min_speed_rate: float, max_speed_rate: float, probability: float
    ):

        self.min_speed_rate = min_speed_rate
        self.max_speed_rate = max_speed_rate
        self.probability = probability

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        r"""Apply the time stretch effect to the waveform.

        Args:
            waveform (Tensor): input waveform
            sample_rate (int): sample rate of the waveform

        Returns:
            Tensor: waveform with the time stretch effect applied
        """

        factor = random.uniform(self.min_speed_rate, self.max_speed_rate)
        effector = torchaudio.io.AudioEffector(effect=f"atempo={factor}", pad_end=False)

        if random.random() < self.probability:
            waveform = effector.apply(waveform.T, sample_rate)
            waveform = waveform.T

        return waveform


class PitchShift:
    r"""Change the pitch of the audio without adjusting the speed.

    Args:
        min_pitch_rate: Minimum pitch rate to change the audio.
        max_pitch_rate: Maximum pitch rate to change the audio.
        probability: Probability of applying the speaker permutation.
    """

    def __init__(
        self, min_pitch_rate: float, max_pitch_rate: float, probability: float
    ):

        self.min_pitch_rate = min_pitch_rate
        self.max_pitch_rate = max_pitch_rate
        self.probability = probability

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        r"""Apply the pitch shift effect to the waveform.

        Args:
            waveform (Tensor): input waveform
            sample_rate (int): sample rate of the waveform

        Returns:
            Tensor: waveform with the pitch shift effect applied
        """

        factor = random.uniform(self.min_pitch_rate, self.max_pitch_rate)
        effector = torchaudio.io.AudioEffector(
            effect=f"asetrate={sample_rate}*{factor},atempo=1/{factor}", pad_end=False
        )

        if random.random() < self.probability:
            waveform = effector.apply(waveform.T, sample_rate)
            waveform = waveform.T

        return waveform
