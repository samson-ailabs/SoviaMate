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

from typing import List, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
import torchaudio
from torch.utils.data import Dataset

from soviamate.utils.helper import load_dataset, stack_batches


class AudioCodecDataset(Dataset):
    r"""The dataset class for training the audio codec model.

    Args:
        filepaths (List[str]): The list of metadata filepaths.
        tokenizer (DictConfig): The tokenizer configuration.
        transforms (DictConfig): The transforms configuration.
    """

    def __init__(
        self, filepaths: List[str], tokenizer: DictConfig, transforms: DictConfig = None
    ):
        super().__init__()

        self.dataset = load_dataset(filepaths)
        self.tokenizer = instantiate(tokenizer)

        if transforms and transforms.get("content") is not None:
            self.content_transforms = [
                instantiate(transform) for transform in transforms.content.values()
            ]

        if transforms and transforms.get("speaker") is not None:
            self.speaker_transforms = [
                instantiate(transform) for transform in transforms.speaker.values()
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]

        audio, sample_rate = torchaudio.load_with_torchcodec(sample["audio_filepath"])
        source_audio = target_audio = prompt_audio = audio.clone()

        # Apply content transforms if any (noise, RIR, etc.)
        if hasattr(self, "content_transforms"):
            for transform in self.content_transforms:
                source_audio = transform.apply(source_audio, sample_rate)

        # Apply speaker transforms if any (trimming, speed perturbation, etc.)
        if hasattr(self, "speaker_transforms"):
            for transform in self.speaker_transforms:
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
