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

"""Utility functions for various tasks"""

import json
import random
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence


def load_dataset(filepaths: Union[str, List[str]]) -> List[Dict[str, Any]]:
    r"""Loads the dataset from the filepaths.

    Args:
        filepaths (Union[str, List[str]]): The filepaths to the dataset.

    Returns:
        List[Dict[str, Any]]
            The dataset loaded from the filepaths.
    """

    if isinstance(filepaths, str):
        filepaths = [filepaths]

    dataset = []
    for filepath in filepaths:
        with open(filepath, mode="r", encoding="utf-8") as f:
            dataset += [json.loads(line) for line in f]

    return dataset


def stack_batches(
    sequences: List[torch.Tensor], padding_value: float = 0.0
) -> torch.Tensor:
    r"""Stacks the input sequences into a batch tensor.

    Args:
        sequences (List[Tensor]): The input sequences to be padded, each with shape
            ``(*, T)`` where ``T`` is the length of the sequence.
        padding_value (float, optional): The padding value. Defaults to 0.0.

    Returns:
        Tensor
            The padded sequences.
    """

    sequences = [seq.t() for seq in sequences]
    lengths = [seq.size(0) for seq in sequences]

    sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    lengths = torch.tensor(lengths, device=sequences.device, dtype=torch.long)

    return sequences, lengths


def make_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    r"""Generates the padding masks.

    Args:
        lengths (Tensor): The length of the input tensors.

    Returns:
        Tensor
            The padding masks. A ``True`` value indicates that
            the corresponding input value will be ignored.
    """

    device = lengths.device
    dtype = lengths.dtype

    batch_size = lengths.size(0)
    max_length = lengths.max()

    mask = torch.arange(max_length, device=device, dtype=dtype)
    mask = mask.expand(batch_size, max_length) >= lengths.unsqueeze(1)

    return mask


def make_attention_mask(
    lengths: torch.Tensor, chunk_size: int, left_context: int
) -> torch.Tensor:
    r"""Generates the chunk-based attention masks.

    Args:
        lengths (Tensor): The length of the input tensors.
        chunk_size (int): The length of each input segment.
        left_context (int): The length of left context.

    Returns:
        Tensor
            The attention masks. A ``True`` value indicates that
            the corresponding input value will be ignored.
    """

    device = lengths.device
    dtype = lengths.dtype

    offset = left_context // chunk_size
    max_length = lengths.max()

    row = torch.arange(max_length, device=device, dtype=dtype)
    row = row.div(chunk_size, rounding_mode="trunc")

    col = torch.arange(max_length + left_context, device=device, dtype=dtype)
    col = col.div(chunk_size, rounding_mode="trunc")

    diff = row.unsqueeze(1) - col.unsqueeze(0) + offset
    mask = torch.gt(diff, offset) | torch.lt(diff, 0)

    return mask


def sample_chunk_config(
    lengths: torch.Tensor,
    min_chunk_size: int = 1,
    max_chunk_size: int = -1,
    left_context_ratio: int = 4,
    full_context_prob: float = 0.0,
) -> Tuple[int, int]:
    r"""Sample chunk configuration for dynamic chunk training.

    Randomly samples chunk size to enable unified streaming and non-streaming models.

    Args:
        lengths (Tensor): Sequence lengths in the batch with shape `(B,)`.
        min_chunk_size (int): Minimum chunk size in frames. Default: 1.
        max_chunk_size (int): Maximum chunk size. If -1, uses max length. Default: -1.
        left_context_ratio (int): Ratio of left_context to chunk_size. Default: 4.
        full_context_prob (float): Probability of full context mode. Default: 0.0.

    Returns:
        Tuple[int, int]: (chunk_size, left_context)
    """

    min_length = int(lengths.min().item())
    max_length = int(lengths.max().item())

    if random.random() >= full_context_prob and min_chunk_size < min_length:
        effective_max_chunk_size = (
            max_length if max_chunk_size < 0 else min(max_chunk_size, max_length)
        )

        chunk_size = random.randint(min_chunk_size, effective_max_chunk_size)
        left_context = left_context_ratio * chunk_size

        if chunk_size + left_context < min_length:
            return chunk_size, left_context

    return max_length, 0
