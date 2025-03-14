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

"""Extracting embeddings using Adapter models"""

from typing import Tuple

import torch
import torch.nn as nn

from soviamate.utils.helper import make_padding_mask


class SpeakerAdapter(nn.Module):
    r"""Speaker Adapter module for conditioning inputs on reference speaker characteristics.

    Args:
        input_dim (int): input feature dimension.
        hidden_dim (int): hidden dimension of the feed-forward module.
        num_heads (int): number of attention heads.
        dropout (float): dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.normalize1 = nn.LayerNorm(input_dim)
        self.normalize2 = nn.LayerNorm(input_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        spk_utt_embs: torch.Tensor,
        spk_frm_embs: torch.Tensor,
        spk_frm_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the adapter module.

        Args:
            inputs (Tensor): input features, shape (B, T, D)
            lengths (Tensor): lengths of input features, shape (B,)
            spk_utt_embs (Tensor): speaker utterance embeddings, shape (B, 1, D)
            spk_frm_embs (Tensor): speaker frame embeddings, shape (B, T', D)
            spk_frm_lens (Tensor): lengths of speaker frame embeddings, shape (B,)

        Returns:
            Tensor: output features, shape (B, T, D)
            Tensor: lengths of output features, shape (B,)
        """

        prompt_masks = make_padding_mask(spk_frm_lens)

        xs, _ = self.attention(
            inputs, spk_frm_embs, spk_frm_embs, key_padding_mask=prompt_masks
        )

        xs = self.normalize1(xs + inputs)
        xs = self.normalize2(xs + self.feedforward(xs))

        xs = xs + spk_utt_embs
        x_lens = lengths.clone()

        return xs, x_lens
