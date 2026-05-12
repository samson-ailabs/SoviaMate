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

"""Evaluate audio codec quality and voice conversion performance.

Modes:
  - reconstruction: PESQ, STOI, UTMOS, WER, SECS on codec round-trip.
  - conversion: SECS, WER, UTMOS on cross-speaker voice conversion.
"""

import argparse
import csv
import json
import logging
import os
import random
import string
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)
from torchmetrics.functional.text import word_error_rate
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from soviamate.bundles import AudioCodecBundle
from soviamate.layers.ecapa_tdnn import ECAPATDNNHead

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("s3prl").setLevel(logging.ERROR)

SAMPLE_RATE = 16000
DEFAULT_SV_CHECKPOINT = "checkpoints/speaker_verification/wavlm_ecapa.pth"


# ---------------------------------------------------------------------------
# Speaker similarity: WavLM-Large + ECAPA-TDNN
# ---------------------------------------------------------------------------


class WavLMSpeakerEncoder(ECAPATDNNHead):
    """WavLM-Large + ECAPA-TDNN speaker encoder (256-dim, cosine similarity).

    Inherits ECAPA-TDNN head so that ``load_state_dict(strict=False)``
    loads both ``feature_extract.*`` and head weights in one call.
    """

    def __init__(self, sv_checkpoint: str, device: str = "cpu"):
        super().__init__()
        self.device = device

        self.feature_extract = torch.hub.load(
            "s3prl/s3prl:main", "wavlm_large", trust_repo=True
        )
        state = torch.load(sv_checkpoint, map_location="cpu", weights_only=True)

        self.load_state_dict(state.get("model", state), strict=False)
        self.to(device).eval()

    @torch.inference_mode()
    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized speaker embedding (256,) from (1, T) audio."""
        audio = waveform.squeeze().float().to(self.device)
        hidden_states = self.feature_extract([audio])["hidden_states"]
        return F.normalize(super().forward(hidden_states), dim=-1).squeeze(0)

    def similarity(self, wav_a: torch.Tensor, wav_b: torch.Tensor) -> float:
        """Cosine similarity between speaker embeddings of two utterances."""
        return torch.dot(self.embed(wav_a), self.embed(wav_b)).item()


# ---------------------------------------------------------------------------
# WER: Whisper-large-v3 ASR
# ---------------------------------------------------------------------------

# Strip all punctuation except apostrophes.
_PUNCTUATION = str.maketrans("", "", string.punctuation.replace("'", ""))


def _normalize_text(text: str) -> str:
    """Normalize text for WER: lowercase, strip punctuation, collapse whitespace."""
    return " ".join(text.lower().translate(_PUNCTUATION).split())


class WhisperASR:
    """Whisper-large-v3 transcriber for WER computation."""

    def __init__(self, device: str = "cpu"):
        model_id = "openai/whisper-large-v3"
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(
            device
        )
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def transcribe(self, waveform: torch.Tensor) -> str:
        """Transcribe (1, T) 16kHz audio to text."""
        audio = waveform.squeeze().float().cpu().numpy()

        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = inputs.input_features.to(device=self.device, dtype=self.model.dtype)

        generated = self.model.generate(inputs, language="english", task="transcribe")
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute utterance-level WER after text normalization."""
    hyp = _normalize_text(hypothesis)
    ref = _normalize_text(reference)
    return 0.0 if not ref else word_error_rate([hyp], [ref]).item()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_manifest(
    manifest_path: str, min_duration: float = 0.0, max_duration: float = float("inf")
) -> list[dict]:
    """Load JSONL manifest with optional duration filtering."""
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            dur = entry.get("duration", 0.0)
            if min_duration <= dur <= max_duration:
                entries.append(entry)
    return entries


def load_conversion_pairs(meta_path: str) -> list[dict]:
    """Load conversion pairs from a pipe-delimited meta file.

    Format::

        name|prompt_text|prompt_wav|source_text|source_wav

    Args:
        meta_path: Path to the meta file.

    Returns:
        List of pair dicts with source/prompt paths and texts.
    """
    meta_dir = str(Path(meta_path).parent)
    pairs = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue

            prompt_wav = parts[2]
            if not os.path.isabs(prompt_wav):
                prompt_wav = os.path.join(meta_dir, prompt_wav)

            source_wav = parts[4]
            if not os.path.isabs(source_wav):
                source_wav = os.path.join(meta_dir, source_wav)

            pairs.append(
                {
                    "name": parts[0],
                    "source_path": source_wav,
                    "source_text": parts[3],
                    "prompt_path": prompt_wav,
                }
            )
    return pairs


def load_audio(path: str, device: torch.device) -> torch.Tensor:
    """Load audio file as (1, T) tensor at 16 kHz."""
    return (
        AudioDecoder(path, sample_rate=SAMPLE_RATE, num_channels=1)
        .get_all_samples()
        .data.to(device)
    )


def _print_results(mode: str, scores: dict[str, list[float]], n_samples: int):
    """Print evaluation results."""
    metric_order = [
        ("secs", "SECS"),
        ("wer", "WER"),
        ("utmos", "UTMOS"),
        ("pesq", "PESQ"),
        ("stoi", "STOI"),
    ]

    print(f"\n{'=' * 54}")
    print(f"EVALUATION RESULTS ({mode})")
    print(f"{'=' * 54}")
    print(f"Evaluated: {n_samples} samples")
    print(f"{'-' * 54}")
    print(f"{'Metric':<14} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-' * 54}")

    for key, label in metric_order:
        values = scores.get(key, [])
        if not values:
            continue

        arr = np.array(values)
        print(
            f"{label:<14} {arr.mean():>8.4f} {arr.std():>8.4f} "
            f"{arr.min():>8.4f} {arr.max():>8.4f}"
        )

    print(f"{'=' * 54}")


def _save_csv(path: str, results: list[dict]):
    """Save per-sample results to CSV."""
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {path}")


def _gather_results(scores, results, world_size):
    """Gather scores and results from all distributed ranks."""
    if world_size <= 1:
        return scores, results

    dist.barrier()

    all_scores = [None] * world_size
    all_results = [None] * world_size

    dist.all_gather_object(all_scores, scores)
    dist.all_gather_object(all_results, results)

    merged_scores = {k: [v for gpu in all_scores for v in gpu[k]] for k in scores}
    merged_results = [r for gpu in all_results for r in gpu]

    return merged_scores, merged_results


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------


def _eval_reconstruction(bundle, entries, args, device, rank, world_size, is_main):
    """Evaluate codec reconstruction: reference-based and reference-free metrics."""
    pesq_metric = PerceptualEvaluationSpeechQuality(SAMPLE_RATE, "wb").to(device)
    stoi_metric = ShortTimeObjectiveIntelligibility(SAMPLE_RATE).to(device)

    utmos_model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )
    utmos_model.to(device).eval()

    asr_model = WhisperASR(device=str(device))
    spk_model = WavLMSpeakerEncoder(args.sv_checkpoint, device=str(device))

    entries = entries[rank::world_size]

    if args.output_audio:
        save_dir = Path(args.output_audio)
        save_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, list[float]] = {
        "pesq": [],
        "stoi": [],
        "utmos": [],
        "wer": [],
        "secs": [],
    }
    results: list[dict] = []

    for entry in tqdm(entries, desc=f"[{rank}] Reconstruction", disable=not is_main):
        try:
            waveform = load_audio(entry["audio_filepath"], device)
            reconstructed, _ = bundle(waveform)

            min_len = min(waveform.size(-1), reconstructed.size(-1))
            ref = waveform[..., :min_len]
            deg = reconstructed[..., :min_len]

            pesq_val = pesq_metric(deg, ref).item()
            stoi_val = stoi_metric(deg, ref).item()
            utmos_val = utmos_model(deg, sr=SAMPLE_RATE).item()
            secs_val = spk_model.similarity(deg, ref)

            ref_text = entry.get("transcript", "")
            if ref_text:
                wer_val = compute_wer(asr_model.transcribe(deg), ref_text)
            else:
                wer_val = None

            if args.output_audio:
                stem = Path(entry["audio_filepath"]).stem
                enc = AudioEncoder(deg.squeeze(0).cpu(), sample_rate=SAMPLE_RATE)
                enc.to_file(str(save_dir / f"{stem}_recon.wav"))

            scores["pesq"].append(pesq_val)
            scores["stoi"].append(stoi_val)
            scores["utmos"].append(utmos_val)
            scores["secs"].append(secs_val)

            if wer_val is not None:
                scores["wer"].append(wer_val)

            results.append(
                {
                    "file": entry["audio_filepath"],
                    "pesq": pesq_val,
                    "stoi": stoi_val,
                    "utmos": utmos_val,
                    "wer": wer_val,
                    "secs": secs_val,
                }
            )

        except Exception as e:
            if is_main:
                print(f"Error: {entry['audio_filepath']}: {e}")
            continue

    scores, results = _gather_results(scores, results, world_size)

    if is_main:
        _print_results("Reconstruction", scores, len(results))
        if args.scores_csv:
            _save_csv(args.scores_csv, results)


def _eval_conversion(bundle, pairs, args, device, rank, world_size, is_main):
    """Evaluate voice conversion: speaker similarity, intelligibility, quality."""
    if is_main:
        print(f"  {len(pairs)} conversion pairs")

    pairs = pairs[rank::world_size]

    utmos_model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )
    utmos_model.to(device).eval()

    spk_model = WavLMSpeakerEncoder(args.sv_checkpoint, device=str(device))
    asr_model = WhisperASR(device=str(device))

    if args.output_audio:
        save_dir = Path(args.output_audio)
        save_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, list[float]] = {"secs": [], "wer": [], "utmos": []}
    results: list[dict] = []

    for pair in tqdm(pairs, desc=f"[{rank}] Conversion", disable=not is_main):
        try:
            source_wav = load_audio(pair["source_path"], device)
            prompt_wav = load_audio(pair["prompt_path"], device)

            converted, _ = bundle(source_wav, prompt_audios=prompt_wav)

            utmos_val = utmos_model(converted, sr=SAMPLE_RATE).item()
            secs_val = spk_model.similarity(converted, prompt_wav)

            ref_text = pair.get("source_text", "")
            if ref_text:
                wer_val = compute_wer(asr_model.transcribe(converted), ref_text)
            else:
                wer_val = None

            scores["utmos"].append(utmos_val)
            scores["secs"].append(secs_val)
            if wer_val is not None:
                scores["wer"].append(wer_val)

            row = {
                "name": pair.get("name", ""),
                "source": pair["source_path"],
                "prompt": pair["prompt_path"],
                "secs": secs_val,
                "wer": wer_val,
                "utmos": utmos_val,
            }

            if args.output_audio:
                name = pair.get("name", Path(pair["source_path"]).stem)
                enc = AudioEncoder(converted.squeeze(0).cpu(), sample_rate=SAMPLE_RATE)
                enc.to_file(str(save_dir / f"{name}.wav"))

            results.append(row)

        except Exception as e:
            if is_main:
                print(f"Error: {pair['source_path']} -> {pair['prompt_path']}: {e}")
            continue

    scores, results = _gather_results(scores, results, world_size)

    if is_main:
        _print_results("Voice Conversion", scores, len(results))
        if args.scores_csv:
            _save_csv(args.scores_csv, results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _setup_distributed():
    """Setup distributed environment and return (device, rank, world_size)."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            device_id = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")

    else:
        rank, world_size = 0, 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return device, rank, world_size


def main():
    """Parse arguments, setup distributed, and run evaluation."""
    torch.set_grad_enabled(False)

    examples = (
        "examples:\n"
        "  # Evaluate codec reconstruction quality\n"
        "  python %(prog)s --checkpoint model.pt reconstruction manifest.jsonl\n"
        "\n"
        "  # Evaluate voice conversion quality\n"
        "  python %(prog)s --checkpoint model.pt conversion pairs.lst\n"
        "\n"
        "  # Distributed evaluation across multiple GPUs\n"
        "  torchrun --nproc_per_node=4 %(prog)s --checkpoint model.pt reconstruction manifest.jsonl\n"
    )

    parser = argparse.ArgumentParser(
        description="Evaluate audio codec quality and voice conversion performance",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Audio codec model checkpoint",
    )
    parser.add_argument(
        "--sv-checkpoint",
        type=str,
        default=DEFAULT_SV_CHECKPOINT,
        help="Speaker verification checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--scores-csv",
        type=str,
        default=None,
        help="CSV path for per-sample scores",
    )
    parser.add_argument(
        "--output-audio",
        type=str,
        default=None,
        help="Directory to save output wav files",
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # -- reconstruction subcommand --
    recon = sub.add_parser("reconstruction", help="Codec round-trip evaluation")
    recon.add_argument(
        "manifest",
        type=str,
        help="JSONL manifest with audio_filepath and duration",
    )
    recon.add_argument(
        "--max-utterances",
        type=int,
        default=None,
        help="Max utterances to evaluate",
    )
    recon.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Min utterance duration (s)",
    )
    recon.add_argument(
        "--max-duration",
        type=float,
        default=float("inf"),
        help="Max utterance duration (s)",
    )

    # -- conversion subcommand --
    conv = sub.add_parser("conversion", help="Voice conversion evaluation")
    conv.add_argument(
        "pairs",
        type=str,
        help="Pipe-delimited file with conversion pairs",
    )

    args = parser.parse_args()
    device, rank, world_size = _setup_distributed()
    is_main = rank == 0

    if is_main:
        print(f"Device: {device} | Processes: {world_size} | Mode: {args.mode}")

    bundle = AudioCodecBundle.from_checkpoint(args.checkpoint, device)

    if args.mode == "reconstruction":
        entries = load_manifest(args.manifest, args.min_duration, args.max_duration)
        if args.max_utterances and args.max_utterances < len(entries):
            entries = random.Random(42).sample(entries, k=args.max_utterances)

        if not entries:
            if is_main:
                print("No entries after filtering.")
            if dist.is_initialized():
                dist.destroy_process_group()
            return

        if is_main:
            durs = [e.get("duration", 0) for e in entries]
            print(
                f"Manifest: {len(entries)} utterances, duration {min(durs):.1f}-{max(durs):.1f}s"
            )

        _eval_reconstruction(bundle, entries, args, device, rank, world_size, is_main)

    else:
        pairs = load_conversion_pairs(args.pairs)
        if is_main:
            print(f"Loaded {len(pairs)} conversion pairs from {args.pairs}")

        _eval_conversion(bundle, pairs, args, device, rank, world_size, is_main)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
