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

        if transforms and transforms.get("source") is not None:
            self.source_transforms = [
                instantiate(transform) for transform in transforms.source.values()
            ]

        if transforms and transforms.get("target") is not None:
            self.target_transforms = [
                instantiate(transform) for transform in transforms.target.values()
            ]

        if transforms and transforms.get("prompt") is not None:
            self.prompt_transforms = [
                instantiate(transform) for transform in transforms.prompt.values()
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]

        source_audio, source_sample_rate = torchaudio.load(sample["audio_filepath"])
        target_audio, target_sample_rate = torchaudio.load(sample["target_filepath"])

        if source_sample_rate != target_sample_rate:
            source_audio = torchaudio.functional.resample(
                source_audio, source_sample_rate, target_sample_rate
            )
            source_sample_rate = target_sample_rate

        # Apply source transforms (degradation augmentation on source domain)
        if hasattr(self, "source_transforms"):
            for transform in self.source_transforms:
                source_audio = transform.apply(source_audio, source_sample_rate)

        # Apply target transforms (speaker/timbre augmentation on target domain)
        if hasattr(self, "target_transforms"):
            for transform in self.target_transforms:
                target_audio = transform.apply(target_audio, target_sample_rate)

        # Create prompt from target audio or load from file
        if sample.get("prompt_filepath") is not None:
            prompt_audio, prompt_sample_rate = torchaudio.load(
                sample["prompt_filepath"]
            )
            assert prompt_sample_rate == target_sample_rate, (
                f"Sample rates mismatch: {prompt_sample_rate} != {target_sample_rate}"
            )
        else:
            # Use transformed target audio as base for prompt
            prompt_audio, prompt_sample_rate = target_audio, target_sample_rate

        # Apply prompt transforms (extraction/trimming from target)
        if hasattr(self, "prompt_transforms"):
            for transform in self.prompt_transforms:
                prompt_audio = transform.apply(prompt_audio, prompt_sample_rate)

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

        source_audios, source_lengths = stack_batches(source_audios)
        prompt_audios, prompt_lengths = stack_batches(prompt_audios)

        target_audios, target_lengths = stack_batches(target_audios)
        target_tokens, target_token_lengths = stack_batches(target_tokens)

        return (
            source_audios,
            source_lengths,
            prompt_audios,
            prompt_lengths,
            target_audios,
            target_lengths,
            target_tokens,
            target_token_lengths,
        )
