#!/usr/bin/env python3
"""
Prepare tokenizer: extract transcripts from JSONL and train SentencePiece tokenizer.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from soviamate.datas.tokenizer import SentencePieceTokenizer


def extract_transcripts(jsonl_file: str, output_file: str) -> int:
    """Extract transcripts from JSONL file.

    Returns:
        Number of valid transcripts extracted.
    """
    processed_count = 0
    valid_count = 0
    total_chars = 0

    print(f"Extracting transcripts from {jsonl_file}")

    with (
        open(jsonl_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                transcript = data.get("transcript", "").strip()

                if transcript and len(transcript) > 10:
                    outfile.write(transcript + "\n")
                    valid_count += 1
                    total_chars += len(transcript)

                processed_count += 1

                if processed_count % 100000 == 0:
                    print(f"Processed: {processed_count:,}, Valid: {valid_count:,}")

            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")
                continue

    print(f"Extracted {valid_count:,} transcripts ({total_chars:,} chars)")
    return valid_count


def train_tokenizer(
    input_file: str,
    model_dir: str,
    model_name: str,
    vocab_size: int = 1024,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
) -> None:
    """Train SentencePiece tokenizer."""
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{model_name}.model"

    print(f"\nTraining tokenizer: {model_file}")
    print(f"Vocab size: {vocab_size}, Type: {model_type}")

    tokenizer = SentencePieceTokenizer(
        model_path=str(model_file),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
    )

    tokenizer.train(
        input_file=input_file,
        model_prefix=str(model_path / model_name),
        split_digits=True,
        treat_whitespace_as_suffix=False,
    )

    print(f"Tokenizer saved to: {model_file}")
    print(f"Actual vocabulary size: {tokenizer.vocab_size_actual}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract transcripts and train tokenizer"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--model-name", required=True, help="Model name")
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

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)

    _, transcript_file = tempfile.mkstemp(suffix=".txt")

    try:
        # Step 1: Extract transcripts
        valid_count = extract_transcripts(args.input, transcript_file)

        if valid_count == 0:
            print("Error: No valid transcripts extracted")
            sys.exit(1)

        # Step 2: Train tokenizer
        train_tokenizer(
            input_file=transcript_file,
            model_dir=args.model_dir,
            model_name=args.model_name,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            model_type=args.model_type,
        )

    finally:
        Path(transcript_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
