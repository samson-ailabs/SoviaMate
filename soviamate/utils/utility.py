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

""" Utility functions for various tasks """

import torch


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
    max_length = int(lengths.max())

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
    max_length = int(lengths.max())

    row = torch.arange(max_length, device=device, dtype=dtype)
    row = row.div(chunk_size, rounding_mode="trunc")

    col = torch.arange(max_length + left_context, device=device, dtype=dtype)
    col = col.div(chunk_size, rounding_mode="trunc")

    diff = row.unsqueeze(1) - col.unsqueeze(0) + offset
    mask = torch.gt(diff, offset) | torch.lt(diff, 0)

    return mask
