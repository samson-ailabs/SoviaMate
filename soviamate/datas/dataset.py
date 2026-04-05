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

import base64
from typing import List, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from soviamate.utils.helper import load_dataset, stack_batches


class AudioCodecDataset(Dataset):
    """Dataset for training the audio codec with speaker disentanglement.

    Args:
        filepaths: Metadata filepaths (one JSONL per split).
        tokenizer: Tokenizer configs for encoding transcripts into tokens.
        transforms: Optional audio transforms for perturbing source audio.
    """

    def __init__(
        self, filepaths: List[str], tokenizer: DictConfig, transforms: DictConfig = None
    ):
        super().__init__()

        self.dataset = load_dataset(filepaths)
        self.tokenizer = instantiate(tokenizer)

        if transforms is not None:
            self.augmentations = [
                instantiate(transform) for transform in transforms.values()
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]

        decoder = AudioDecoder(
            sample["audio_filepath"], sample_rate=16000, num_channels=1
        )

        target_audio = decoder.get_all_samples().data
        sample_rate = decoder.metadata.sample_rate

        # Augment source audio for content encoding
        source_audio = target_audio.clone()
        if hasattr(self, "augmentations"):
            for transform in self.augmentations:
                source_audio = transform.apply(source_audio, sample_rate)

        # Decode speaker embedding from base64-encoded float32
        speaker_embedding = bytearray(base64.b64decode(sample["speaker_embedding"]))
        speaker_embedding = torch.frombuffer(speaker_embedding, dtype=torch.float32)

        # Encode transcript for text supervision
        target_tokens = self.tokenizer.encode(sample["transcript"])
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        return source_audio, speaker_embedding, target_audio, target_tokens

    @staticmethod
    def collate_data(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        """Pad and batch the per-sample 4-tuples into a 7-element batch."""

        source_audios, speaker_embeddings, target_audios, target_tokens = zip(*batch)

        source_audios, source_audio_lengths = stack_batches(source_audios)
        speaker_embeddings = torch.stack(speaker_embeddings)

        target_audios, target_audio_lengths = stack_batches(target_audios)
        target_tokens, target_token_lengths = stack_batches(target_tokens)

        return (
            source_audios,
            source_audio_lengths,
            speaker_embeddings,
            target_audios,
            target_audio_lengths,
            target_tokens,
            target_token_lengths,
        )
