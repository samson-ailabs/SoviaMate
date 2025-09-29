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

"""Tokenizer implementations for conversational speech ASR"""

import os
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm  # type: ignore[import-untyped]


class SentencePieceTokenizer:
    r"""SentencePiece tokenizer optimized for conversational speech with proper nouns.

    Preserves index 0 as a special blank token for CTC-style ASR decoders.

    Args:
        model_path (str): Path to the trained SentencePiece model file (.model)
        vocab_size (int, optional): Vocabulary size for training. Defaults to 8000.
        character_coverage (float, optional): Character coverage for training. Defaults to 1.0.
        model_type (str, optional): Model type ('bpe', 'unigram', 'char', 'word'). Defaults to 'bpe'.
    """

    def __init__(
        self,
        model_path: str,
        vocab_size: int = 8000,
        character_coverage: float = 1.0,
        model_type: str = "bpe",
    ):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.model_type = model_type

        # Initialize processor
        self.sp = spm.SentencePieceProcessor()

        # Load model if it exists
        if os.path.exists(model_path):
            self.sp.load(model_path)
        else:
            print(f"Warning: SentencePiece model not found at {model_path}")
            print("Call train() method to create the model first.")

    def train(
        self,
        input_file: str,
        model_prefix: Optional[str] = None,
        split_digits: bool = True,
        treat_whitespace_as_suffix: bool = False,
    ) -> None:
        """Train SentencePiece model on conversational data.

        Args:
            input_file (str): Path to training text file
            model_prefix (str, optional): Prefix for output model files. If None, uses model_path stem.
            split_digits (bool): Whether to split digits. Defaults to True.
            treat_whitespace_as_suffix (bool): Whitespace handling. Defaults to False.
        """
        if model_prefix is None:
            model_prefix = Path(self.model_path).stem

        # Training arguments optimized for conversational speech
        train_args = [
            f"--input={input_file}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={self.vocab_size}",
            f"--character_coverage={self.character_coverage}",
            f"--model_type={self.model_type}",
            f"--split_digits={split_digits}",
            f"--treat_whitespace_as_suffix={treat_whitespace_as_suffix}",
            "--pad_id=0",  # Reserve 0 for blank/padding
            "--unk_id=1",  # Unknown token
            "--bos_id=2",  # Beginning of sequence
            "--eos_id=3",  # End of sequence
            "--max_sentence_length=16384",  # Handle long conversations
            "--shuffle_input_sentence=true",
            "--normalization_rule_name=identity",  # Preserve original text
        ]

        # Train the model
        try:
            spm.SentencePieceTrainer.train(" ".join(train_args))
        except Exception as e:
            raise RuntimeError(f"SentencePiece training failed: {e}")

        # Load the trained model
        try:
            self.sp.load(f"{model_prefix}.model")
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}")

        # Update model_path to the actual trained model
        self.model_path = f"{model_prefix}.model"

        print(f"SentencePiece model trained and saved to {self.model_path}")
        print(f"Vocabulary size: {self.sp.vocab_size()}")

    @property
    def vocab_size_actual(self) -> int:
        """The actual size of the vocabulary"""
        if hasattr(self.sp, "vocab_size"):
            return self.sp.vocab_size()
        return self.vocab_size

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        """Encode text string into token ids.

        Args:
            text (str): Input text
            add_bos (bool): Add beginning of sequence token
            add_eos (bool): Add end of sequence token

        Returns:
            List[int]: Token IDs (0 is reserved as blank token)
        """
        # SentencePiece encode returns 0-based indices, we shift by +1 to reserve 0 for blank
        tokens = self.sp.encode(text, add_bos=add_bos, add_eos=add_eos)
        return [token + 1 for token in tokens]

    def decode(self, tokens: List[int]) -> str:
        """Decode token ids back to text string.

        Args:
            tokens (List[int]): Token IDs

        Returns:
            str: Decoded text
        """
        # Shift back to 0-based indices for SentencePiece
        sp_tokens = [
            max(0, token - 1) for token in tokens if token > 0
        ]  # Filter out blank tokens
        return self.sp.decode(sp_tokens)

    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode text into subword pieces (for debugging/analysis)."""
        return self.sp.encode_as_pieces(text)

    def id_to_piece(self, token_id: int) -> str:
        """Convert token ID to piece string."""
        if token_id == 0:
            return "<blank>"
        return self.sp.id_to_piece(token_id - 1)

    def piece_to_id(self, piece: str) -> int:
        """Convert piece string to token ID."""
        if piece == "<blank>":
            return 0
        return self.sp.piece_to_id(piece) + 1
