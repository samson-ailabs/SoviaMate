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

"""Dataset modules for preprocessing and augmenting speech data"""

import random
from typing import List, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from soviamate.utils.helper import load_dataset, stack_batches


class AudioCodecDataset(Dataset):
    r"""The dataset class for training the audio codec model.

    Args:
        filepaths (List[str]): The list of metadata filepaths.
        tokenizer (DictConfig): The tokenizer configuration.
        transforms (DictConfig): The transforms configuration.
        prompt_ratio_min (float): Minimum ratio for prompt segment. Defaults to 0.3.
        prompt_ratio_max (float): Maximum ratio for prompt segment. Defaults to 0.5.
        min_prompt_duration (float): Minimum duration for prompt segment. Defaults to 1.0.
    """

    def __init__(
        self,
        filepaths: List[str],
        tokenizer: DictConfig,
        transforms: DictConfig = None,
        prompt_ratio_min: float = 0.3,
        prompt_ratio_max: float = 0.5,
        min_prompt_duration: float = 1.0,
    ):
        super().__init__()

        self.dataset = load_dataset(filepaths)
        self.tokenizer = instantiate(tokenizer)

        self.prompt_ratio_min = prompt_ratio_min
        self.prompt_ratio_max = prompt_ratio_max
        self.min_prompt_duration = min_prompt_duration

        if transforms and transforms.get("source") is not None:
            self.source_transforms = [
                instantiate(transform) for transform in transforms.source.values()
            ]

        if transforms and transforms.get("prompt") is not None:
            self.prompt_transforms = [
                instantiate(transform) for transform in transforms.prompt.values()
            ]

    def _split_audio(
        self, audio: torch.Tensor, sample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split audio into non-overlapping content and prompt segments.

        Args:
            audio: Audio tensor of shape (1, T).
            sample_rate: Audio sample rate.

        Returns:
            content_audio: Second portion for source/target.
            prompt_audio: First portion for speaker conditioning.
        """
        total_samples = audio.shape[1]
        min_prompt_samples = int(self.min_prompt_duration * sample_rate)

        # Sample dynamic prompt ratio between min and max
        prompt_ratio = random.uniform(self.prompt_ratio_min, self.prompt_ratio_max)
        prompt_samples = int(total_samples * prompt_ratio)

        # Ensure prompt meets minimum duration
        prompt_samples = max(prompt_samples, min_prompt_samples)
        prompt_samples = min(prompt_samples, total_samples - min_prompt_samples)

        # If audio is too short, use minimum prompt duration
        if prompt_samples < min_prompt_samples:
            prompt_samples = min(min_prompt_samples, total_samples // 2)

        # Split into non-overlapping segments
        prompt_audio = audio[:, :prompt_samples]
        content_audio = audio[:, prompt_samples:]

        return content_audio, prompt_audio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]

        # Decode audio samples from file
        decoder = AudioDecoder(
            sample["audio_filepath"], sample_rate=16000, num_channels=1
        )

        signal = decoder.get_all_samples()
        audio, sample_rate = signal.data, signal.sample_rate

        # Split audio into content and prompt segments
        content_audio, prompt_audio = self._split_audio(audio, sample_rate)
        target_audio = content_audio.clone()

        # Apply source transforms if any
        source_audio = content_audio.clone()
        if hasattr(self, "source_transforms"):
            for transform in self.source_transforms:
                source_audio = transform.apply(source_audio, sample_rate)

        # Apply prompt transforms if any
        if hasattr(self, "prompt_transforms"):
            for transform in self.prompt_transforms:
                prompt_audio = transform.apply(prompt_audio, sample_rate)

        target_tokens = self.tokenizer.encode(sample["transcript"])
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        return source_audio, prompt_audio, target_audio, target_tokens

    @staticmethod
    def collate_data(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        r"""Collate function for the dataset.

        Args:
            batch (List[Tuple[Tensor, ...]]): The batch of data.

        Returns:
            Tuple[Tensor, ...]: The collated batch.
        """

        source_audios, prompt_audios, target_audios, target_tokens = zip(*batch)

        source_audios, source_audio_lengths = stack_batches(source_audios)
        prompt_audios, prompt_audio_lengths = stack_batches(prompt_audios)

        target_audios, target_audio_lengths = stack_batches(target_audios)
        target_tokens, target_token_lengths = stack_batches(target_tokens)

        return (
            source_audios,
            source_audio_lengths,
            prompt_audios,
            prompt_audio_lengths,
            target_audios,
            target_audio_lengths,
            target_tokens,
            target_token_lengths,
        )
