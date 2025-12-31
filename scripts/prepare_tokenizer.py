#!/usr/bin/env python3
"""
Prepare tokenizer: extract transcripts from JSONL and train SentencePiece tokenizer.
"""

import argparse
import json
import logging
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from soviamate.datas.tokenizer import SentencePieceTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Constants
MIN_TRANSCRIPT_LENGTH = 10
LOG_INTERVAL = 100_000
ENCODING = "utf-8"


# ============================================================================
# Data structures
# ============================================================================


class ExtractionStats(NamedTuple):
    """Statistics from transcript extraction."""

    processed: int
    valid: int
    skipped: int
    total_chars: int


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer preparation."""

    input_files: list[Path]
    model_dir: Path
    model_name: str
    vocab_size: int = 1024
    model_type: str = "bpe"
    character_coverage: float = 0.9995
    normalize: bool = False


# ============================================================================
# Normalization
# ============================================================================


def normalize_transcript(text: str) -> str:
    """Normalize English ASR transcript for training.

    Args:
        text: Raw transcript text

    Returns:
        Normalized transcript text
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove speech annotations and parenthetical remarks
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)

    # Remove URLs and email addresses
    text = re.sub(r"http\S+|www\S+|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", "", text)

    # Remove accents/diacritics
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    # Remove special characters, normalize whitespace, and strip
    text = re.sub(r"[^a-z0-9\s\-']", "", text)

    return " ".join(text.split())


# ============================================================================
# Extraction
# ============================================================================


def extract_transcripts(
    jsonl_files: list[Path], output_file: Path, normalize: bool = True
) -> ExtractionStats:
    """Extract and normalize transcripts from multiple JSONL files.

    Args:
        jsonl_files: List of paths to input JSONL files
        output_file: Path to output transcript file
        normalize: Whether to normalize transcripts

    Returns:
        ExtractionStats with processing results

    Raises:
        ValueError: If any input file doesn't exist
    """
    for jsonl_file in jsonl_files:
        if not jsonl_file.exists():
            raise ValueError(f"Input file not found: {jsonl_file}")

    logger.info("Extracting transcripts from %d file(s)", len(jsonl_files))
    logger.info("Normalization: %s", "enabled" if normalize else "disabled")

    processed, valid, skipped, total_chars = 0, 0, 0, 0

    with open(output_file, "w", encoding=ENCODING) as outfile:
        for jsonl_file in jsonl_files:
            logger.info("Processing: %s", jsonl_file)
            with open(jsonl_file, "r", encoding=ENCODING) as infile:
                for line_num, line in enumerate(infile, 1):
                    try:
                        data = json.loads(line.strip())
                        transcript = data.get("transcript", "").strip()

                        if not transcript:
                            skipped += 1
                            processed += 1
                            continue

                        text = normalize_transcript(transcript) if normalize else transcript

                        if len(text) >= MIN_TRANSCRIPT_LENGTH:
                            outfile.write(text + "\n")
                            valid += 1
                            total_chars += len(text)
                        else:
                            skipped += 1

                        processed += 1

                        if processed % LOG_INTERVAL == 0:
                            logger.info(
                                "Progress: processed=%s, valid=%s, skipped=%s",
                                processed,
                                valid,
                                skipped,
                            )

                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Skipping invalid JSON at %s:%s: %s", jsonl_file.name, line_num, e
                        )
                        continue

    stats = ExtractionStats(processed, valid, skipped, total_chars)
    logger.info(
        "Extraction complete: processed=%s, valid=%s, skipped=%s, chars=%s",
        stats.processed,
        stats.valid,
        stats.skipped,
        stats.total_chars,
    )

    return stats


# ============================================================================
# Training
# ============================================================================


def train_tokenizer(config: TokenizerConfig, input_file: Path) -> None:
    """Train SentencePiece tokenizer.

    Args:
        config: TokenizerConfig with training parameters
        input_file: Path to training data file
    """
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_file = config.model_dir / f"{config.model_name}.model"

    logger.info("\nTraining tokenizer: %s", model_file)
    logger.info("Vocab size: %s, Type: %s", config.vocab_size, config.model_type)

    tokenizer = SentencePieceTokenizer(
        model_path=str(model_file),
        vocab_size=config.vocab_size,
        character_coverage=config.character_coverage,
        model_type=config.model_type,
    )

    tokenizer.train(
        input_file=str(input_file),
        model_prefix=str(config.model_dir / config.model_name),
        split_digits=True,
        treat_whitespace_as_suffix=False,
    )

    logger.info("✓ Tokenizer saved to: %s", model_file)
    logger.info("  Vocabulary size: %s", tokenizer.vocab_size_actual)


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Extract transcripts and train tokenizer"
    )
    parser.add_argument(
        "--input", "-i", type=Path, nargs="+", required=True, help="Input JSONL file(s)"
    )
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("tokenizers"), help="Model directory"
    )
    parser.add_argument("--vocab-size", type=int, default=1024, help="Vocabulary size")
    parser.add_argument(
        "--character-coverage", type=float, default=0.9995, help="Character coverage"
    )
    parser.add_argument(
        "--model-type",
        default="bpe",
        choices=["bpe", "unigram", "char", "word"],
        help="Model type",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply transcript normalization",
    )

    args = parser.parse_args()

    try:
        config = TokenizerConfig(
            input_files=args.input,
            model_dir=args.model_dir,
            model_name=args.model_name,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            normalize=args.normalize,
        )

        # Create temporary file for extracted transcripts
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding=ENCODING
        ) as tmp:
            transcript_file = Path(tmp.name)

        try:
            # Extract transcripts from all input files
            stats = extract_transcripts(
                config.input_files, transcript_file, config.normalize
            )

            if stats.valid == 0:
                logger.error("Error: No valid transcripts extracted")
                return 1

            # Train tokenizer
            train_tokenizer(config, transcript_file)
            return 0

        finally:
            transcript_file.unlink(missing_ok=True)

    except ValueError as e:
        logger.error("Error: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
