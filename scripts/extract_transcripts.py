#!/usr/bin/env python3
"""
Extract transcripts from large JSONL dataset for SentencePiece training.
Handles large files efficiently using streaming processing.
"""

import argparse
import json
import random
from pathlib import Path


def extract_transcripts_streaming(
    jsonl_file: str,
    output_file: str,
    max_lines: int | None = None,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> None:
    """Extract transcripts from JSONL file efficiently.

    Args:
        jsonl_file: Path to input JSONL file
        output_file: Path to output text file for transcripts
        max_lines: Maximum number of lines to process (None for all)
        min_duration: Minimum audio duration to include
        max_duration: Maximum audio duration to include
    """

    processed_count = 0
    valid_count = 0
    total_chars = 0

    print(f"Extracting transcripts from {jsonl_file}")
    print(f"Output file: {output_file}")
    print(f"Duration filter: {min_duration}s - {max_duration}s")

    with (
        open(jsonl_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())

                # Extract transcript and duration
                transcript = data.get("transcript", "").strip()
                duration = data.get("duration", 0)

                # Filter by duration and transcript quality
                if (
                    duration >= min_duration
                    and duration <= max_duration
                    and transcript
                    and len(transcript) > 10
                ):  # At least 10 characters
                    # Write transcript to output file
                    outfile.write(transcript + "\n")
                    valid_count += 1
                    total_chars += len(transcript)

                processed_count += 1

                # Progress reporting
                if processed_count % 100000 == 0:
                    print(
                        f"Processed: {processed_count:,} lines, "
                        f"Valid: {valid_count:,} transcripts"
                    )

                # Stop if max_lines reached
                if max_lines and processed_count >= max_lines:
                    break

            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")
                continue
            except KeyboardInterrupt:
                print(f"\nInterrupted by user at line {line_num}")
                break

    print("\nExtraction complete!")
    print(f"Total processed: {processed_count:,} lines")
    print(f"Valid transcripts: {valid_count:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average transcript length: {total_chars / valid_count:.1f} chars")


def sample_transcripts(
    jsonl_file: str, output_file: str, sample_size: int = 1000000
) -> None:
    """Create a smaller sample for quick testing."""

    print(f"Creating sample of {sample_size:,} lines...")

    # First pass: count total lines (we already know it's 50M)
    total_lines = 50000000

    # Generate random line numbers to sample
    sample_lines = set(random.sample(range(1, total_lines + 1), sample_size))

    processed = 0
    written = 0

    with (
        open(jsonl_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(infile, 1):
            if line_num in sample_lines:
                try:
                    data = json.loads(line.strip())
                    transcript = data.get("transcript", "").strip()

                    if transcript and len(transcript) > 10:
                        outfile.write(transcript + "\n")
                        written += 1

                except json.JSONDecodeError:
                    continue

            processed += 1
            if processed % 1000000 == 0:
                print(f"Processed: {processed:,} lines, Sampled: {written:,}")

    print(f"Sample complete: {written:,} transcripts written")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract transcripts from JSONL dataset"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output text file")
    parser.add_argument("--max-lines", type=int, help="Maximum lines to process")
    parser.add_argument(
        "--min-duration", type=float, default=1.0, help="Minimum duration"
    )
    parser.add_argument(
        "--max-duration", type=float, default=30.0, help="Maximum duration"
    )
    parser.add_argument("--sample", type=int, help="Create random sample of N lines")

    args = parser.parse_args()

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.sample:
        sample_transcripts(args.input, args.output, args.sample)
    else:
        extract_transcripts_streaming(
            args.input,
            args.output,
            args.max_lines,
            args.min_duration,
            args.max_duration,
        )


if __name__ == "__main__":
    main()
