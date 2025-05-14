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

        if transforms and transforms.get("audio") is not None:
            self.audio_transforms = [
                instantiate(transform) for transform in transforms.audio.values()
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

        input_audio, input_sample_rate = torchaudio.load(sample["audio_filepath"])
        target_audio, target_sample_rate = torchaudio.load(sample["target_filepath"])

        assert (
            input_sample_rate == target_sample_rate
        ), f"Sample rates mismatch: {input_sample_rate} != {target_sample_rate}"

        if hasattr(self, "audio_transforms"):
            for transform in self.audio_transforms:
                input_audio = transform.apply(input_audio, input_sample_rate)

        if sample.get("prompt_filepath") is not None:
            prompt_audio, prompt_sample_rate = torchaudio.load(
                sample["prompt_filepath"]
            )
            assert (
                prompt_sample_rate == target_sample_rate
            ), f"Sample rates mismatch: {prompt_sample_rate} != {target_sample_rate}"
        else:
            prompt_audio, prompt_sample_rate = target_audio, target_sample_rate

        if hasattr(self, "prompt_transforms"):
            for transform in self.prompt_transforms:
                prompt_audio = transform.apply(prompt_audio, prompt_sample_rate)

        target_tokens = self.tokenizer.encode(sample["transcript"])
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        return input_audio, prompt_audio, target_audio, target_tokens

    @staticmethod
    def collate_data(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        r"""Collate function for the dataset.

        Args:
            batch (List[Tuple[Tensor, ...]]): The batch of data.

        Returns:
            Tuple[Tensor, ...]: The collated batch.
        """

        input_audios, prompt_audios, target_audios, target_tokens = zip(*batch)

        input_audios, input_lengths = stack_batches(input_audios)
        prompt_audios, prompt_lengths = stack_batches(prompt_audios)

        target_audios, target_lengths = stack_batches(target_audios)
        token_indices, token_lengths = stack_batches(target_tokens)

        return (
            input_audios,
            input_lengths,
            prompt_audios,
            prompt_lengths,
            target_audios,
            target_lengths,
            token_indices,
            token_lengths,
        )
