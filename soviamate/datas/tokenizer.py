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

"""Tokenizer for use with OpenAI's models"""

from typing import List

import tiktoken


class Tokenizer:
    r"""Wrapper for OpenAI's tiktoken that preserves index 0 as a special blank token.

    Args:
        model_name (str, optional): The name of the model to use for tokenization.
        encoding_name (str, optional): The name of the encoding to use for tokenization.
    """

    def __init__(self, model_name: str = None, encoding_name: str = None):
        assert (
            model_name is not None or encoding_name is not None
        ), "Either model_name or encoding_name must be provided"

        assert (
            model_name is None or encoding_name is None
        ), "Only one of model_name or encoding_name can be provided"

        if model_name is not None:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        if encoding_name is not None:
            self.tokenizer = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        r"""The size of the vocabulary"""
        return self.tokenizer.n_vocab + 1

    def encode(self, text: str) -> List[int]:
        r"""Encode a text string into a list of token ids"""
        return [idx + 1 for idx in self.tokenizer.encode(text)]

    def decode(self, tokens: List[int]) -> str:
        r"""Decode a list of token ids into a text string"""
        return self.tokenizer.decode([idx - 1 for idx in tokens])
