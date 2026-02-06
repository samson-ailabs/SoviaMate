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

import numpy as np
import parselmouth
import torch
import torchaudio
from torch.nn import functional as F
from torchaudio import functional as AF

from soviamate.utils.helper import load_dataset


class SpectralPerturbation:
    """STFT-based spectral perturbation for content-speaker disentanglement.

    Applies random biquad filters (low-shelving, high-shelving, peaking) in the
    frequency domain to modify spectral characteristics while preserving content.

    Args:
        window_size (int): STFT window size in samples.
        hop_length (int): STFT hop length in samples.
        cutoff_low (float): Low-shelf filter cutoff frequency in Hz.
        cutoff_high (float): High-shelf filter cutoff frequency in Hz.
        num_peak (int): Number of peaking EQ bands between the shelf filters.
        q_min (float): Minimum Q factor (bandwidth control).
        q_max (float): Maximum Q factor (bandwidth control).
        gain_min (float): Minimum gain in dB (can be negative for cut).
        gain_max (float): Maximum gain in dB (can be negative for cut).
        probability (float): Probability of applying the transform.
    """

    _DB_TO_AMP: float = np.log(10) / 40.0

    def __init__(
        self,
        window_size: int = 1024,
        hop_length: int = 256,
        cutoff_low: float = 60.0,
        cutoff_high: float = 10000.0,
        num_peak: int = 8,
        q_min: float = 2.0,
        q_max: float = 5.0,
        gain_min: float = -12.0,
        gain_max: float = 12.0,
        probability: float = 1.0,
    ):
        self.window_size = window_size
        self.hop_length = hop_length
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.num_peak = num_peak
        self.q_min = q_min
        self.q_max = q_max
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.probability = probability

        # Pre-compute log-spaced peak center frequencies (excluding shelf endpoints)
        indices = torch.arange(1, num_peak + 1, dtype=torch.float32) / (num_peak + 1)
        self.peak_centers = cutoff_low * (cutoff_high / cutoff_low) ** indices

    def _biquad(
        self, iir_coeffs: torch.Tensor, fir_coeffs: torch.Tensor
    ) -> torch.Tensor:
        """Compute frequency response of a biquad filter.

        Args:
            iir_coeffs (Tensor): Feedback coefficients of shape (..., 3).
            fir_coeffs (Tensor): Feedforward coefficients of shape (..., 3).

        Returns:
            Tensor: Complex frequency response of shape (..., window_size // 2 + 1).
        """
        iir = torch.fft.rfft(iir_coeffs, self.window_size, dim=-1)
        fir = torch.fft.rfft(fir_coeffs, self.window_size, dim=-1)
        return fir / iir

    def _low_shelving(
        self,
        cutoff: float,
        gain_db: torch.Tensor,
        q_factor: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """Compute low-shelving filter frequency response.

        Args:
            cutoff (float): Cutoff frequency in Hz.
            gain_db (Tensor): Gain in dB of shape (batch,).
            q_factor (Tensor): Q factor of shape (batch,).
            sample_rate (int): Audio sample rate in Hz.

        Returns:
            Tensor: Complex frequency response of shape (batch, window_size // 2 + 1).
        """
        omega = 2.0 * np.pi * cutoff / sample_rate
        sin_omega, cos_omega = np.sin(omega), np.cos(omega)

        alpha = sin_omega / (2.0 * q_factor)
        cos_omega_t = torch.full_like(q_factor, cos_omega)
        amp = (gain_db * self._DB_TO_AMP).exp()
        sqrt_amp = amp.sqrt()

        # FIR coefficients (numerator)
        b0 = amp * ((amp + 1) - (amp - 1) * cos_omega_t + 2 * sqrt_amp * alpha)
        b1 = 2 * amp * ((amp - 1) - (amp + 1) * cos_omega_t)
        b2 = amp * ((amp + 1) - (amp - 1) * cos_omega_t - 2 * sqrt_amp * alpha)

        # IIR coefficients (denominator)
        a0 = (amp + 1) + (amp - 1) * cos_omega_t + 2 * sqrt_amp * alpha
        a1 = -2 * ((amp - 1) + (amp + 1) * cos_omega_t)
        a2 = (amp + 1) + (amp - 1) * cos_omega_t - 2 * sqrt_amp * alpha

        return self._biquad(
            torch.stack([a0, a1, a2], dim=-1), torch.stack([b0, b1, b2], dim=-1)
        )

    def _high_shelving(
        self,
        cutoff: float,
        gain_db: torch.Tensor,
        q_factor: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """Compute high-shelving filter frequency response.

        Args:
            cutoff (float): Cutoff frequency in Hz.
            gain_db (Tensor): Gain in dB of shape (batch,).
            q_factor (Tensor): Q factor of shape (batch,).
            sample_rate (int): Audio sample rate in Hz.

        Returns:
            Tensor: Complex frequency response of shape (batch, window_size // 2 + 1).
        """
        omega = 2.0 * np.pi * cutoff / sample_rate
        sin_omega, cos_omega = np.sin(omega), np.cos(omega)

        alpha = sin_omega / (2.0 * q_factor)
        cos_omega_t = torch.full_like(q_factor, cos_omega)
        amp = (gain_db * self._DB_TO_AMP).exp()
        sqrt_amp = amp.sqrt()

        # FIR coefficients (numerator)
        b0 = amp * ((amp + 1) + (amp - 1) * cos_omega_t + 2 * sqrt_amp * alpha)
        b1 = -2 * amp * ((amp - 1) + (amp + 1) * cos_omega_t)
        b2 = amp * ((amp + 1) + (amp - 1) * cos_omega_t - 2 * sqrt_amp * alpha)

        # IIR coefficients (denominator)
        a0 = (amp + 1) - (amp - 1) * cos_omega_t + 2 * sqrt_amp * alpha
        a1 = 2 * ((amp - 1) - (amp + 1) * cos_omega_t)
        a2 = (amp + 1) - (amp - 1) * cos_omega_t - 2 * sqrt_amp * alpha

        return self._biquad(
            torch.stack([a0, a1, a2], dim=-1), torch.stack([b0, b1, b2], dim=-1)
        )

    def _peaking_eq(
        self,
        center_freq: torch.Tensor,
        gain_db: torch.Tensor,
        q_factor: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """Compute peaking equalizer frequency response.

        Args:
            center_freq (Tensor): Center frequency in Hz of shape (batch,) or (batch, num_peak).
            gain_db (Tensor): Gain in dB of shape (batch,) or (batch, num_peak).
            q_factor (Tensor): Q factor of shape (batch,) or (batch, num_peak).
            sample_rate (int): Audio sample rate in Hz.

        Returns:
            Tensor: Complex frequency response of shape (batch, window_size // 2 + 1)
                or (batch, num_peak, window_size // 2 + 1).
        """
        omega = 2.0 * np.pi * center_freq / sample_rate
        sin_omega, cos_omega = torch.sin(omega), torch.cos(omega)

        alpha = sin_omega / (2.0 * q_factor)
        amp = (gain_db * self._DB_TO_AMP).exp()

        return self._biquad(
            torch.stack([1 + alpha / amp, -2 * cos_omega, 1 - alpha / amp], dim=-1),
            torch.stack([1 + alpha * amp, -2 * cos_omega, 1 - alpha * amp], dim=-1),
        )

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply spectral perturbation to audio waveform.

        Args:
            waveform (Tensor): Audio waveform of shape (1, num_samples).
            sample_rate (int): Audio sample rate in Hz.

        Returns:
            Tensor: Filtered waveform of shape (1, num_samples), normalized to [-1, 1].
        """
        if random.random() > self.probability:
            return waveform

        num_samples = waveform.shape[1]

        # Compute STFT
        window = torch.hann_window(self.window_size)
        spec = torch.stft(
            waveform,
            n_fft=self.window_size,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=window,
            return_complex=True,
        )

        # Sample random parameters for all filters
        num_filters = self.num_peak + 2  # peaks + low shelf + high shelf
        power = torch.rand(1, num_filters)

        q_factor = self.q_min * (self.q_max / self.q_min) ** power
        gain_db = torch.empty(1, num_filters).uniform_(self.gain_min, self.gain_max)

        # Compute peaking filters: [1, num_peak, F] -> [1, F]
        centers = self.peak_centers.unsqueeze(0)
        peak_response = self._peaking_eq(
            centers, gain_db[:, :-2], q_factor[:, :-2], sample_rate
        )
        peak_response = torch.prod(peak_response, dim=1)

        # Compute shelf filters: [1, F]
        low_response = self._low_shelving(
            self.cutoff_low, gain_db[:, -2], q_factor[:, -2], sample_rate
        )
        high_response = self._high_shelving(
            self.cutoff_high, gain_db[:, -1], q_factor[:, -1], sample_rate
        )

        # Apply combined filter in frequency domain
        response = peak_response * low_response * high_response
        spec_filtered = spec * response.unsqueeze(-1)

        # Inverse STFT
        output = torch.istft(
            spec_filtered,
            n_fft=self.window_size,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=window,
            length=num_samples,
        )

        return output.clamp(-1.0, 1.0)


class VoicePerturbation:
    """PSOLA-based voice perturbation for content-speaker disentanglement.

    Randomly transforms formants, pitch, and pitch dynamics to remove speaker
    identity while preserving linguistic content.

    Args:
        formant_shift (float): Max formant shift factor. Samples from [1/f, f].
        pitch_shift (float): Max pitch shift factor. Samples from [1/p, p].
        pitch_range (float): Max pitch range factor. Samples from [1/r, r].
        pitch_steps (float): F0 analysis time step in seconds.
        pitch_floor (float): Minimum F0 for pitch detection in Hz.
        pitch_ceiling (float): Maximum F0 for pitch detection in Hz.
        probability (float): Probability of applying the transform.
    """

    def __init__(
        self,
        formant_shift: float = 1.4,
        pitch_shift: float = 1.5,
        pitch_range: float = 1.4,
        pitch_steps: float = 0.01,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 600.0,
        probability: float = 1.0,
    ):
        self.formant_shift = formant_shift
        self.pitch_shift = pitch_shift
        self.pitch_range = pitch_range
        self.pitch_steps = pitch_steps
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.probability = probability

    def _sample_ratio(self, max_ratio: float) -> float:
        """Sample ratio symmetrically from [1/max_ratio, max_ratio]."""
        ratio = random.random() * (max_ratio - 1.0) + 1.0
        return ratio if random.random() > 0.5 else 1.0 / ratio

    def apply(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply voice perturbation to audio waveform.

        Args:
            waveform (Tensor): Audio waveform of shape (1, num_samples).
            sample_rate (int): Audio sample rate in Hz.

        Returns:
            Tensor: Perturbed waveform of shape (1, num_samples).
        """
        if random.random() > self.probability:
            return waveform

        waveform_np = waveform.squeeze(0).numpy()

        # Sample random perturbation factors
        formant_shift = self._sample_ratio(self.formant_shift)
        pitch_shift = self._sample_ratio(self.pitch_shift)
        pitch_range = self._sample_ratio(self.pitch_range)

        try:
            sound = parselmouth.Sound(waveform_np, sampling_frequency=sample_rate)
            pitch = parselmouth.praat.call(
                sound,
                "To Pitch",
                self.pitch_steps,
                self.pitch_floor,
                self.pitch_ceiling,
            )

            f0_array = pitch.selected_array["frequency"]
            voiced_f0 = f0_array[f0_array > 1e-5]
            if len(voiced_f0) == 0:
                return waveform

            f0_median = np.median(voiced_f0).item()
            new_f0_median = f0_median * pitch_shift

            # Safeguard against Praat hang bug (github.com/praat/praat/issues/1926)
            f0_min = voiced_f0.min().item()
            scaled_min = (
                new_f0_median + (f0_min * pitch_shift - new_f0_median) * pitch_range
            )
            if scaled_min < 0.0:
                pitch_range = 1.0

            transformed = parselmouth.praat.call(
                [sound, pitch],
                "Change gender",
                formant_shift,
                new_f0_median,
                pitch_range,
                1.0,  # duration_factor
            )

            waveform_np = transformed.values.squeeze(0)

        except Exception:
            pass

        return torch.from_numpy(waveform_np).unsqueeze(0).float()


class NoiseInjection:
    r"""Inject background noise into the audio signal.

    Args:
        noise_filepaths (str): List of metadata filepaths for the noise samples.
        min_amplitude (float): Minimum amplitude of the noise.
        max_amplitude (float): Maximum amplitude of the noise.
        probability (float): Probability of applying the noise injection.
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

        assert noise_signal.norm() > 1e-6, (
            f"Background noise signal is empty: {noise_filepath}"
        )

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
        rir_filepaths (str): List of metadata filepaths for the impulse response samples.
        probability (float): Probability of applying the impulse response.
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

        assert rir_signal.norm() > 1e-6, (
            f"Impulse response signal is empty: {rir_filepath}"
        )

        if rir_sample_rate != sample_rate:
            rir_signal = AF.resample(rir_signal, rir_sample_rate, sample_rate)

        waveform = AF.fftconvolve(waveform, rir_signal, mode="same")

        return waveform


class SpecAugment:
    r"""Apply the SpecAugment effect to the audio signal.

    Args:
        num_freq_mask (int): Number of frequency masks to apply.
        freq_mask_width (int): Width of the frequency mask.
        num_time_mask (int): Number of time masks to apply.
        time_mask_width (int): Width of the time mask.
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
