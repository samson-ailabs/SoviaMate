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

r"""Modules for Transducer-based Automatic Speech Recognition"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from soviamate.utils.helper import make_padding_mask


class Predictor(nn.Module):
    r"""A stateless prediction network for RNN-T model.

    Args:
        embedding_dim (int): embedding dimension for the input tokens.
        hidden_dim (int): hidden dimension of the network.
        num_embeddings (int): number of tokens in the vocabulary.
        context_size (int): size of history context for convolution.
        dropout (float): dropout probability for the convolution layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_embeddings: int,
        context_size: int,
        dropout: float,
    ) -> None:

        super().__init__()
        self.blank_token = 0

        self.num_embeddings = num_embeddings
        self.context_size = context_size

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.normalize = nn.LayerNorm(embedding_dim)

        self.network = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, context_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for training the model.

        Args:
            tokens (Tensor): target sequences with shape `(B, U)`.
            lengths (Tensor): lengths of the target sequences with shape `(B,)`.

        Returns:
            Tuple[Tensor, Tensor]: output sequences with shape `(B, U + 1, D)`
                and their lengths with shape `(B,)`.
        """

        tokens = F.pad(tokens, (self.context_size, 0), value=self.blank_token)
        lengths = lengths + self.context_size

        embedding_out = self.embedding(tokens)
        embedding_out = self.normalize(embedding_out)

        masks = make_padding_mask(lengths)[:, :, None]
        embedding_out = embedding_out.masked_fill(masks, 0.0)

        conv_inps = embedding_out.permute(0, 2, 1)
        conv_outs = self.network(conv_inps)

        outputs = conv_outs.permute(0, 2, 1)
        lengths = lengths - self.context_size + 1

        return outputs, lengths

    @torch.jit.export
    def infer(self, contexts: torch.Tensor) -> torch.Tensor:
        r"""Forward pass for streaming inference.

        Args:
            contexts (Tensor): history context with shape `(B, S)`.

        Returns:
            Tensor: output sequences with shape `(B, 1, D)`.
        """

        embedding_out = self.embedding(contexts)
        embedding_out = self.normalize(embedding_out)

        conv_inps = embedding_out.permute(0, 2, 1)
        conv_outs = self.network(conv_inps)

        outputs = conv_outs.permute(0, 2, 1)

        return outputs


class Joint(nn.Module):
    r"""A joint network for RNN-T model.

    Args:
        encoder_hidden (int): hidden dimension of the encoder.
        predictor_hidden (int): hidden dimension of the predictor.
        joint_hidden (int): hidden dimension of the joint network.
    """

    def __init__(
        self, encoder_hidden: int, predictor_hidden: int, joint_hidden: int
    ) -> None:

        super().__init__()

        self.enc_proj = nn.Linear(encoder_hidden, joint_hidden)
        self.pred_proj = nn.Linear(predictor_hidden, joint_hidden)

    def forward(
        self, encoder_outputs: torch.Tensor, predictor_outputs: torch.Tensor
    ) -> torch.Tensor:
        r"""Forward pass for training the model.

        Args:
            encoder_outputs (Tensor): encoder outputs with shape `(B, T, D1)`.
            predictor_outputs (Tensor): predictor outputs with shape `(B, U, D2)`.

        Returns:
            Tensor: joint outputs with shape `(B, T, U, D)`.
        """

        enc_outs = self.enc_proj(encoder_outputs)
        pred_outs = self.pred_proj(predictor_outputs)

        outputs = enc_outs.unsqueeze(2) + pred_outs.unsqueeze(1)
        outputs = F.relu(outputs.contiguous())

        return outputs
