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
from torchcodec.decoders import AudioDecoder

from soviamate.utils.helper import load_dataset, stack_batches


class AudioCodecDataset(Dataset):
    """Dataset for training the audio codec with speaker disentanglement.

    Args:
        filepaths: Metadata filepaths (one JSONL per split).
        tokenizer: Tokenizer configuration.
        transforms: ``source`` and ``prompt`` transform configs.
    """

    SAMPLE_RATE: int = 16000
    NUM_MELS: int = 80

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

        if transforms and transforms.get("prompt") is not None:
            self.prompt_transforms = [
                instantiate(transform) for transform in transforms.prompt.values()
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]

        decoder = AudioDecoder(
            sample["audio_filepath"], sample_rate=self.SAMPLE_RATE, num_channels=1
        )
        audio = decoder.get_all_samples().data

        # Target: full utterance for reconstruction
        target_audio = audio.clone()

        # Source: perturbed for content encoding
        source_audio = audio.clone()
        if hasattr(self, "source_transforms"):
            for transform in self.source_transforms:
                source_audio = transform.apply(source_audio, self.SAMPLE_RATE)

        # Fbank: full utterance for speaker embedding
        fbank = torchaudio.compliance.kaldi.fbank(
            audio, num_mel_bins=self.NUM_MELS, sample_frequency=self.SAMPLE_RATE
        )
        prompt_fbank = (fbank - fbank.mean(dim=0, keepdim=True)).t()

        # Prompt: augmented waveform for mel conditioning
        prompt_audio = audio.clone()
        if hasattr(self, "prompt_transforms"):
            for transform in self.prompt_transforms:
                prompt_audio = transform.apply(prompt_audio, self.SAMPLE_RATE)

        # Target tokens: transcript for content supervision
        target_tokens = self.tokenizer.encode(sample["transcript"])
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        return source_audio, prompt_audio, prompt_fbank, target_audio, target_tokens

    @staticmethod
    def collate_data(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        r"""Pad and batch the per-sample 5-tuples into a 10-element batch."""

        source_audios, prompt_audios, prompt_fbanks, target_audios, target_tokens = zip(
            *batch
        )

        source_audios, source_audio_lengths = stack_batches(source_audios)

        prompt_audios, prompt_audio_lengths = stack_batches(prompt_audios)
        prompt_fbanks, prompt_fbank_lengths = stack_batches(prompt_fbanks)

        target_audios, target_audio_lengths = stack_batches(target_audios)
        target_tokens, target_token_lengths = stack_batches(target_tokens)

        return (
            source_audios,
            source_audio_lengths,
            prompt_audios,
            prompt_audio_lengths,
            prompt_fbanks,
            prompt_fbank_lengths,
            target_audios,
            target_audio_lengths,
            target_tokens,
            target_token_lengths,
        )
