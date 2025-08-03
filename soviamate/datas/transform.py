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
from torch.nn import functional as F
from torchaudio import functional as AF

from soviamate.utils.helper import load_dataset


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
            waveform (Tensor): input waveform, shape (1, L)
            sample_rate (int): sample rate of the waveform

        Returns:
            Tensor: waveform with the time stretch effect applied
        """

        if random.random() > self.probability:
            return waveform

        factor = random.uniform(self.min_speed_rate, self.max_speed_rate)
        effector = torchaudio.io.AudioEffector(effect=f"atempo={factor}", pad_end=False)
        waveform = effector.apply(waveform.T, sample_rate)

        return waveform.T


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
            waveform (Tensor): input waveform, shape (1, L)
            sample_rate (int): sample rate of the waveform

        Returns:
            Tensor: waveform with the pitch shift effect applied
        """

        if random.random() > self.probability:
            return waveform

        factor = random.uniform(self.min_pitch_rate, self.max_pitch_rate)
        effector = torchaudio.io.AudioEffector(
            effect=f"asetrate={sample_rate}*{factor},atempo=1/{factor}", pad_end=False
        )
        waveform = effector.apply(waveform.T, sample_rate)

        return waveform.T


class AudioTrimmer:
    r"""Trim the audio to a fixed percentage of the original length.

    Args:
        percentage: Percentage of the original length to keep.
        min_duration: Minimum duration of the audio after trimming.
        probability: Probability of applying the audio trimming.
    """

    def __init__(self, percentage: float, min_duration: float, probability: float):
        self.percentage = percentage
        self.min_duration = min_duration
        self.probability = probability

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        r"""Trim the waveform to the maximum length.

        Args:
            waveform (Tensor): input waveform, shape (1, L)
            sample_rate (int): sample rate of the waveform

        Returns:
            Tensor: waveform with the trimming effect applied
        """

        if random.random() > self.probability:
            return waveform

        duration = waveform.size(1) / sample_rate * self.percentage
        duration = max(self.min_duration, duration)

        max_length = int(duration * sample_rate)
        max_length = min(max_length, waveform.size(1))

        if max_length > 0:
            start = random.randint(0, waveform.size(1) - max_length)
            waveform = waveform[:, start : start + max_length]

        return waveform


class NoiseInjection:
    r"""Inject background noise into the audio signal.

    Args:
        noise_filepaths: List of metadata filepaths for the noise samples.
        min_amplitude: Minimum amplitude of the noise.
        max_amplitude: Maximum amplitude of the noise.
        probability: Probability of applying the noise injection.
    """

    def __init__(
        self,
        noise_filepaths: str,
        min_amplitude: float,
        max_amplitude: float,
        probability: float,
    ):

        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.probability = probability

        self.noise_samples = load_dataset(noise_filepaths)

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        r"""Inject noise into the waveform.

        Args:
            waveform (Tensor): input waveform, shape (1, L)
            sample_rate (int): sample rate of the waveform

        Returns:
            Tensor: waveform with the noise injected
        """

        if random.random() > self.probability:
            return waveform

        noise_filepath = random.choice(self.noise_samples)
        noise_signal, noise_sample_rate = torchaudio.load(noise_filepath)

        assert (
            noise_signal.norm() > 1e-6
        ), f"Background noise signal is empty: {noise_filepath}"

        if noise_sample_rate != sample_rate:
            noise_signal = AF.resample(noise_signal, noise_sample_rate, sample_rate)

        audio_length = waveform.size(1)
        noise_length = noise_signal.size(1)

        offset = abs(audio_length - noise_length)
        index = random.randint(0, offset)

        if audio_length >= noise_length:
            noise_signal = F.pad(noise_signal, (index, offset - index))
        else:
            noise_signal = noise_signal[:, index : index + audio_length]

        snr_db = random.uniform(self.min_amplitude, self.max_amplitude)
        waveform = AF.add_noise(waveform, noise_signal, torch.tensor([snr_db]))

        return waveform


class ImpulseResponse:
    r"""Apply the impulse response effect to the audio signal.

    Args:
        rir_filepaths: List of metadata filepaths for the impulse response samples.
        probability: Probability of applying the impulse response.
    """

    def __init__(self, rir_filepaths: str, probability: float):
        self.probability = probability
        self.rir_samples = load_dataset(rir_filepaths)

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        r"""Apply the impulse response effect to the waveform.

        Args:
            waveform (Tensor): input waveform, shape (1, L)
            sample_rate (int): sample rate of the waveform

        Returns:
            Tensor: waveform with the impulse response effect applied
        """

        if random.random() > self.probability:
            return waveform

        rir_filepath = random.choice(self.rir_samples)
        rir_signal, rir_sample_rate = torchaudio.load(rir_filepath)

        assert (
            rir_signal.norm() > 1e-6
        ), f"Impulse response signal is empty: {rir_filepath}"

        if rir_sample_rate != sample_rate:
            rir_signal = AF.resample(rir_signal, rir_sample_rate, sample_rate)

        waveform = AF.fftconvolve(waveform, rir_signal, mode="same")

        return waveform


class SpecAugment:
    r"""Apply the SpecAugment effect to the audio signal.

    Args:
        num_freq_mask: Number of frequency masks to apply.
        freq_mask_width: Width of the frequency mask.
        num_time_mask: Number of time masks to apply.
        time_mask_width: Width of the time mask.
    """

    def __init__(
        self,
        num_freq_mask: int,
        freq_mask_width: int,
        num_time_mask: int,
        time_mask_width: int,
    ):
        self.num_freq_mask = num_freq_mask
        self.freq_mask_width = freq_mask_width
        self.num_time_mask = num_time_mask
        self.time_mask_width = time_mask_width

    def apply(self, spectrogram: torch.Tensor) -> torch.Tensor:
        r"""Apply the SpecAugment effect to the waveform.

        Args:
            spectrogram (Tensor): input spectrogram, shape (B, F, T)

        Returns:
            Tensor: spectrogram with the SpecAugment effect applied
        """

        _, num_freqs, num_frames = spectrogram.size()

        freq_width = max(1, int(self.freq_mask_width * num_freqs))
        time_width = max(1, int(self.time_mask_width * num_frames))

        for _ in range(self.num_freq_mask):
            spectrogram = AF.mask_along_axis(spectrogram, freq_width, 0.0, 1)

        for _ in range(self.num_time_mask):
            spectrogram = AF.mask_along_axis(spectrogram, time_width, 0.0, 2)

        return spectrogram
