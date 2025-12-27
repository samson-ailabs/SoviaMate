#!/usr/bin/env python3
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

"""Evaluate audio codec quality using PESQ and STOI on LibriSpeech test-clean."""

import argparse
import csv
import os
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from torchcodec.decoders import AudioDecoder
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)
from tqdm import tqdm

from soviamate.bundles import AudioCodecBundle

SAMPLE_RATE = 16000


def load_librispeech_test_clean(
    data_dir: str, max_samples: int | None = None, rank: int = 0, world_size: int = 1
) -> Iterator[tuple[torch.Tensor, str]]:
    """Load LibriSpeech test-clean samples, sharded across GPUs."""
    test_clean_dir = Path(data_dir) / "test-clean"

    if not test_clean_dir.exists():
        raise FileNotFoundError(f"LibriSpeech test-clean not found at {test_clean_dir}")

    flac_files = sorted(test_clean_dir.rglob("*.flac"))

    if max_samples is not None:
        flac_files = flac_files[:max_samples]

    # Shard files across GPUs
    flac_files = flac_files[rank::world_size]

    for flac_path in flac_files:
        decoder = AudioDecoder(str(flac_path), sample_rate=SAMPLE_RATE, num_channels=1)
        signal = decoder.get_all_samples()

        yield signal.data, str(flac_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio codec on LibriSpeech")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    parser.add_argument("--data-dir", type=str, required=True, help="LibriSpeech root")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
    parser.add_argument("--output", type=str, default=None, help="Output CSV")
    args = parser.parse_args()

    # Setup distributed
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank, world_size = 0, 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    if is_main:
        print(f"Loading model from {args.checkpoint}...")
        print(f"Running on {world_size} GPU(s)")

    bundle = AudioCodecBundle.from_checkpoint(args.checkpoint, device)

    if is_main:
        print(f"Loading LibriSpeech test-clean from {args.data_dir}...")

    samples = list(
        load_librispeech_test_clean(args.data_dir, args.max_samples, rank, world_size)
    )

    # Initialize metrics
    pesq_metric = PerceptualEvaluationSpeechQuality(SAMPLE_RATE, "wb").to(device)
    stoi_metric = ShortTimeObjectiveIntelligibility(SAMPLE_RATE).to(device)

    pesq_scores, stoi_scores, results = [], [], []

    for waveform, file_path in tqdm(samples, desc=f"GPU {rank}", disable=not is_main):
        waveform = waveform.to(device)

        # Reconstruct through codec
        reconstructed, _ = bundle(waveform)

        # Align lengths
        min_len = min(waveform.size(-1), reconstructed.size(-1))
        ref = waveform[..., :min_len]
        deg = reconstructed[..., :min_len]

        try:
            pesq_score = pesq_metric(deg, ref).item()
            stoi_score = stoi_metric(deg, ref).item()
        except Exception as e:
            if is_main:
                print(f"Error processing {file_path}: {e}")
            continue

        pesq_scores.append(pesq_score)
        stoi_scores.append(stoi_score)

        results.append({"file": file_path, "pesq": pesq_score, "stoi": stoi_score})

    # Gather results from all GPUs
    if world_size > 1:
        all_pesq = [None] * world_size
        all_stoi = [None] * world_size
        all_results = [None] * world_size

        dist.all_gather_object(all_pesq, pesq_scores)
        dist.all_gather_object(all_stoi, stoi_scores)
        dist.all_gather_object(all_results, results)

        pesq_scores = [s for gpu_scores in all_pesq for s in gpu_scores]
        stoi_scores = [s for gpu_scores in all_stoi for s in gpu_scores]
        results = [r for gpu_results in all_results for r in gpu_results]

    # Print results (only on main process)
    if is_main:
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Evaluated: {len(pesq_scores)} samples")
        print("-" * 50)

        if pesq_scores:
            print(
                f"PESQ: {sum(pesq_scores) / len(pesq_scores):.3f} "
                f"(min: {min(pesq_scores):.3f}, max: {max(pesq_scores):.3f})"
            )

        if stoi_scores:
            print(
                f"STOI: {sum(stoi_scores) / len(stoi_scores):.3f} "
                f"(min: {min(stoi_scores):.3f}, max: {max(stoi_scores):.3f})"
            )

        print("=" * 50)

        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["file", "pesq", "stoi"])
                writer.writeheader()
                writer.writerows(results)
            print(f"\nResults saved to {args.output}")

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
