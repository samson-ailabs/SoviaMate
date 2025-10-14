#!/usr/bin/env python3
"""
Train SentencePiece tokenizer on conversational speech data.
"""

import argparse
from pathlib import Path
import sys

from soviamate.datas.tokenizer import SentencePieceTokenizer


def train_tokenizer(
    input_file: str,
    model_dir: str = "models",
    model_name: str = "asr_tokenizer",
    vocab_size: int = 1024,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
) -> None:
    """Train SentencePiece tokenizer on conversation data.

    Args:
        input_file: Path to training text file (one sentence per line)
        model_dir: Directory to save the model
        model_name: Name for the model files
        vocab_size: Vocabulary size
        character_coverage: Character coverage (1.0 for full coverage)
        model_type: Model type (bpe, unigram, char, word)
    """

    # Create model directory
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # Full path to model
    model_file = model_path / f"{model_name}.model"

    print("Training SentencePiece tokenizer...")
    print(f"Input file: {input_file}")
    print(f"Model file: {model_file}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Character coverage: {character_coverage}")
    print(f"Model type: {model_type}")

    # Initialize tokenizer
    tokenizer = SentencePieceTokenizer(
        model_path=str(model_file),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
    )

    # Train the tokenizer
    tokenizer.train(
        input_file=input_file,
        model_prefix=str(model_path / model_name),
        split_digits=True,  # Good for phone numbers, addresses
        treat_whitespace_as_suffix=False,
    )

    print("\nTokenizer training complete!")
    print(f"Model saved to: {model_file}")

    # Test the tokenizer
    test_sentences = [
        "Hello, my name is John Smith from New York.",
        "Can you help me with the authentication system?",
        "The server is running on localhost:8080",
        "I'm experiencing issues with the McDonald's WiFi connection.",
        "Please call me at 555-123-4567 or email john.doe@example.com",
    ]

    print("\nTesting tokenizer with sample sentences:")
    print("=" * 60)

    for sentence in test_sentences:
        tokens = tokenizer.encode(sentence)
        pieces = tokenizer.encode_as_pieces(sentence)
        decoded = tokenizer.decode(tokens)

        print(f"Original: {sentence}")
        print(f"Tokens:   {tokens}")
        print(f"Pieces:   {pieces}")
        print(f"Decoded:  {decoded}")
        print(
            f"Vocab coverage: {len([p for p in pieces if not p.startswith('▁')]) / len(pieces) * 100:.1f}%"
        )
        print("-" * 60)

    print(f"\nActual vocabulary size: {tokenizer.vocab_size_actual}")


def analyze_coverage(input_file: str, model_file: str, num_samples: int = 1000) -> None:
    """Analyze tokenizer coverage on sample data."""

    print("Analyzing tokenizer coverage...")

    # Load tokenizer
    tokenizer = SentencePieceTokenizer(model_path=model_file)

    total_chars = 0
    total_tokens = 0
    unk_tokens = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            text = line.strip()
            if not text:
                continue

            tokens = tokenizer.encode(text)
            pieces = tokenizer.encode_as_pieces(text)

            total_chars += len(text)
            total_tokens += len(tokens)
            unk_tokens += pieces.count("<unk>")

    print(f"Coverage analysis on {num_samples} samples:")
    print(f"Average compression ratio: {total_chars / total_tokens:.2f} chars/token")
    print(f"Unknown token rate: {unk_tokens / total_tokens * 100:.2f}%")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument("--input", "-i", required=True, help="Input text file")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--model-name", default="asr_tokenizer", help="Model name")
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
        "--analyze", action="store_true", help="Run coverage analysis after training"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)

    # Train tokenizer
    train_tokenizer(
        input_file=args.input,
        model_dir=args.model_dir,
        model_name=args.model_name,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
    )

    # Run analysis if requested
    if args.analyze:
        model_file = Path(args.model_dir) / f"{args.model_name}.model"
        analyze_coverage(args.input, str(model_file))


if __name__ == "__main__":
    main()
