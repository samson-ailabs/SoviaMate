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

"""Audio Codec Bundle for production."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence, unpad_sequence


@dataclass
class CodecInputs:
    """Structured inputs for AudioCodecBundle.

    Attributes:
        source_audios: Batched audio tensors (B, T, 1).
        source_audio_lengths: Length of each audio (B,).
        prompt_audios: Optional speaker prompts (B, T_prompt, 1).
        prompt_audio_lengths: Optional prompt lengths (B,).
        prompt_fbanks: Optional fbank features (B, T', n_mels).
        prompt_fbank_lengths: Optional fbank lengths (B,).
    """

    source_audios: torch.Tensor
    source_audio_lengths: torch.Tensor
    prompt_audios: Optional[torch.Tensor] = None
    prompt_audio_lengths: Optional[torch.Tensor] = None
    prompt_fbanks: Optional[torch.Tensor] = None
    prompt_fbank_lengths: Optional[torch.Tensor] = None


@dataclass
class CodecOutputs:
    """Structured outputs from AudioCodecBundle.

    Attributes:
        audios: Decoded audio.
        audio_lengths: Length of each audio.
        tokens: Optional ASR tokens.
        token_lengths: Optional token lengths.
    """

    audios: Union[torch.Tensor, List[torch.Tensor]]
    audio_lengths: Optional[torch.Tensor]
    tokens: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    token_lengths: Optional[torch.Tensor] = None


class AudioCodecBundle(nn.Module):
    """Audio Codec Bundle for production.

    This class bundles the audio encoder, quantizer, and decoder along with
    optional ASR decoder and speaker adapter for voice conversion. Loads from
    checkpoints created by AudioCodecTask.export_model() for production deployment.

    Args:
        audio_encoder: The audio encoder module.
        audio_quantizer: The audio quantizer module.
        audio_decoder: The audio decoder module with cross-attention.
        text_decoder: Optional ASR text decoder module.
        speaker_adapter: Optional speaker adapter module for voice conversion.
        device: Device to place the model on ('cpu', 'cuda', 'cuda:0', etc.).
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        audio_quantizer: nn.Module,
        audio_decoder: nn.Module,
        text_decoder: Optional[nn.Module] = None,
        speaker_adapter: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        self.n_mels = 80  # Must match speaker adapter config
        self.sample_rate = 16000  # Must match speaker adapter config

        self.audio_encoder = audio_encoder
        self.audio_quantizer = audio_quantizer
        self.audio_decoder = audio_decoder
        self.text_decoder = text_decoder
        self.speaker_adapter = speaker_adapter

        if device is not None:
            self.to(device)

        # Set to eval mode for inference
        self.eval()

    @property
    def device(self) -> torch.device:
        """Get the device the model is currently on.

        Returns:
            Device object representing the model's device.
        """
        return next(self.parameters()).device

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str, device: Optional[Union[str, torch.device]] = None
    ) -> "AudioCodecBundle":
        """Load AudioCodecBundle from exported checkpoint.

        Args:
            checkpoint: Path to checkpoint created by AudioCodecTask.export_model().
            device: Device to load the model to ('cpu', 'cuda', 'cuda:0', etc.).

        Returns:
            AudioCodecBundle instance with loaded weights.
        """
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint (exported with plain dicts, safe to load)
        device = device if device is not None else "cpu"
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Extract model weights (must be from export_model)
        if "model_weights" not in checkpoint:
            raise KeyError(
                "Checkpoint must contain `model_weights` key. "
                "Use AudioCodecTask.export_model() to create a valid checkpoint."
            )

        model_weights = checkpoint["model_weights"]

        # Extract component weights
        audio_encoder_weights = model_weights.get("audio_encoder", {})
        audio_quantizer_weights = model_weights.get("audio_quantizer", {})
        audio_decoder_weights = model_weights.get("audio_decoder", {})
        text_decoder_weights = model_weights.get("text_decoder", {})
        speaker_adapter_weights = model_weights.get("speaker_adapter", {})

        # Instantiate components from hyperparameters
        if "hyper_parameters" not in checkpoint:
            raise KeyError(
                "Checkpoint must contain `hyper_parameters` key. "
                "Use AudioCodecTask.export_model() to create a valid checkpoint."
            )

        hparams = checkpoint["hyper_parameters"]
        model_config = hparams.get("model", hparams)

        audio_encoder = instantiate(model_config["audio_encoder"])
        audio_quantizer = instantiate(model_config["audio_quantizer"])
        audio_decoder = instantiate(model_config["audio_decoder"])

        text_decoder = None
        if "text_decoder" in model_config and text_decoder_weights:
            text_decoder = instantiate(model_config["text_decoder"])

        speaker_adapter = None
        if "speaker_adapter" in model_config and speaker_adapter_weights:
            speaker_adapter = instantiate(model_config["speaker_adapter"])

        # Load model weights
        audio_encoder.load_state_dict(audio_encoder_weights)
        audio_quantizer.load_state_dict(audio_quantizer_weights)
        audio_decoder.load_state_dict(audio_decoder_weights)

        if text_decoder is not None and text_decoder_weights:
            text_decoder.load_state_dict(text_decoder_weights)

        if speaker_adapter is not None and speaker_adapter_weights:
            speaker_adapter.load_state_dict(speaker_adapter_weights)

        # Build bundle
        bundle = cls(
            audio_encoder=audio_encoder,
            audio_quantizer=audio_quantizer,
            audio_decoder=audio_decoder,
            text_decoder=text_decoder,
            speaker_adapter=speaker_adapter,
            device=device,
        )

        return bundle

    @staticmethod
    def pack_sequence(
        audio_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack list of variable-length audio tensors.

        Args:
            audio_list: List of audio tensors with shape (1, T_i)
                where T_i is the time dimension for each audio.

        Returns:
            audios: Padded batch tensor (B, T_max, 1).
            lengths: Original sequence lengths tensor (B,).
        """
        if not audio_list:
            raise ValueError("Cannot pack empty audio list")

        audios = [audio.t() for audio in audio_list]
        lengths = [audio.size(1) for audio in audio_list]

        audios = pad_sequence(audios, batch_first=True, padding_value=0.0)
        lengths = torch.tensor(lengths, dtype=torch.long, device=audios.device)

        return audios, lengths

    @staticmethod
    def unpack_sequence(
        audios: torch.Tensor, lengths: torch.Tensor
    ) -> List[torch.Tensor]:
        """Unpack batched tensor into list of variable-length tensors.

        Args:
            audios: Batched audio tensor (B, T_max, 1).
            lengths: Original sequence lengths tensor (B,).

        Returns:
            List of audio tensors with shape (1, T_i) for each sequence.
        """
        audios = unpad_sequence(audios, lengths, batch_first=True)
        return [audio.t() for audio in audios]

    def _pack_inputs(
        self,
        source_audios: Union[torch.Tensor, List[torch.Tensor]],
        prompt_audios: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    ) -> CodecInputs:
        """Pack flexible input formats to batched structure.

        Args:
            source_audios: Single audio tensor or list of audio tensors.
            prompt_audios: Optional speaker prompts matching source format.

        Returns:
            CodecInputs with batched tensors ready for processing.
        """
        if isinstance(source_audios, list):
            source_audios, source_audio_lengths = self.pack_sequence(source_audios)
        else:
            source_audios = source_audios.t().unsqueeze(0)
            source_audio_lengths = torch.tensor(
                [source_audios.size(1)], device=source_audios.device
            )

        prompt_audio_lengths = None
        batch_size = source_audios.size(0)

        if prompt_audios is not None:
            if isinstance(prompt_audios, list):
                if len(prompt_audios) != batch_size:
                    raise ValueError(
                        f"Number of prompts ({len(prompt_audios)}) must match "
                        f"number of audios ({batch_size})"
                    )
                prompt_audios, prompt_audio_lengths = self.pack_sequence(prompt_audios)
            else:
                prompt_audios = prompt_audios.t().unsqueeze(0)
                prompt_audios = prompt_audios.expand(batch_size, -1, -1)

                seq_length, device = prompt_audios.size(1), prompt_audios.device
                prompt_audio_lengths = torch.full(
                    (batch_size,), seq_length, dtype=torch.long, device=device
                )

        # Compute fbank features for speaker adapter
        prompt_fbanks = None
        prompt_fbank_lengths = None

        if prompt_audios is not None and self.speaker_adapter is not None:
            prompt_fbanks, prompt_fbank_lengths = self._compute_fbank(
                prompt_audios, prompt_audio_lengths
            )

        return CodecInputs(
            source_audios=source_audios,
            source_audio_lengths=source_audio_lengths,
            prompt_audios=prompt_audios,
            prompt_audio_lengths=prompt_audio_lengths,
            prompt_fbanks=prompt_fbanks,
            prompt_fbank_lengths=prompt_fbank_lengths,
        )

    def _compute_fbank(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute kaldi fbank features with cepstral mean normalization.

        Args:
            audios: Batched waveform of shape (B, T, 1).
            audio_lengths: Actual lengths in samples of shape (B,).

        Returns:
            fbanks: Padded fbank features of shape (B, T', n_mels).
            fbank_lengths: Feature lengths of shape (B,).
        """
        batch_size = audios.size(0)
        device = audios.device

        fbanks, fbank_lengths = [], []
        for i in range(batch_size):
            length = audio_lengths[i].item()
            wav = audios[i, :length, 0].unsqueeze(0)

            fbank = torchaudio.compliance.kaldi.fbank(
                wav, num_mel_bins=self.n_mels, sample_frequency=self.sample_rate
            )
            fbank = fbank - fbank.mean(dim=0, keepdim=True)

            fbanks.append(fbank)
            fbank_lengths.append(fbank.size(0))

        max_len = max(fbank_lengths)
        padded_fbank = torch.zeros(
            batch_size, max_len, self.n_mels, device=device, dtype=audios.dtype
        )
        for i, fbank in enumerate(fbanks):
            padded_fbank[i, : fbank.size(0)] = fbank

        fbank_lengths = torch.tensor(fbank_lengths, dtype=torch.long, device=device)
        return padded_fbank, fbank_lengths

    def _process(
        self, codec_inputs: CodecInputs, return_tokens: bool = False
    ) -> CodecOutputs:
        """Process audio through the codec pipeline with optional speaker conditioning.

        Args:
            codec_inputs: Structured inputs with batched tensors.
            return_tokens: Whether to return ASR tokens.

        Returns:
            CodecOutputs with processed audio and optional tokens.
        """
        # Validate optional components
        if return_tokens and self.text_decoder is None:
            raise ValueError(
                "Cannot return tokens: text_decoder not available. "
                "Load checkpoint with text_decoder component."
            )

        if codec_inputs.prompt_audios is not None and self.speaker_adapter is None:
            raise ValueError(
                "Cannot apply speaker adaptation: speaker_adapter not available. "
                "Load checkpoint with speaker_adapter component."
            )

        # Encode source audio
        source_features, source_feature_lengths = self.audio_encoder(
            codec_inputs.source_audios, codec_inputs.source_audio_lengths
        )

        # ASR decoding (optional)
        output_tokens = None
        token_lengths = None

        if return_tokens:
            output_tokens, token_lengths = self.text_decoder(
                source_features, source_feature_lengths
            )

        # Quantize
        quantized_outputs, quantized_lengths = self.audio_quantizer(
            source_features, source_feature_lengths
        )

        # Extract speaker features from prompt (optional)
        speaker_features = None
        speaker_feature_lengths = None

        if self.speaker_adapter is not None and codec_inputs.prompt_audios is not None:
            speaker_features, speaker_feature_lengths = self.speaker_adapter(
                codec_inputs.prompt_audios,
                codec_inputs.prompt_audio_lengths,
                codec_inputs.prompt_fbanks,
                codec_inputs.prompt_fbank_lengths,
            )

        # Calculate maximum output length for exact reconstruction
        max_output_length = codec_inputs.source_audio_lengths.max().item()

        # Decode with speaker conditioning
        output_audios, output_lengths = self.audio_decoder(
            quantized_outputs,
            quantized_lengths,
            speaker_features,
            speaker_feature_lengths,
            max_output_length,
        )

        return CodecOutputs(
            audios=output_audios,
            audio_lengths=output_lengths,
            tokens=output_tokens,
            token_lengths=token_lengths,
        )

    def _unpack_outputs(
        self,
        codec_outputs: CodecOutputs,
        as_list: bool = False,
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Optional[Union[torch.Tensor, List[torch.Tensor]]],
    ]:
        """Unpack outputs to requested format.

        Args:
            codec_outputs: Batched outputs from process().
            as_list: Whether to return as list.

        Returns:
            Tuple of (audios, tokens). If as_list is True, returns lists of
            tensors. If False, returns single tensors (first element).
        """
        audio_list = self.unpack_sequence(
            codec_outputs.audios, codec_outputs.audio_lengths
        )

        token_list = None
        if codec_outputs.tokens is not None:
            token_list = self.unpack_sequence(
                codec_outputs.tokens, codec_outputs.token_lengths
            )

        if as_list:
            return audio_list, token_list

        return audio_list[0], token_list[0] if token_list else None

    @torch.inference_mode()
    def forward(
        self,
        source_audios: Union[torch.Tensor, List[torch.Tensor]],
        prompt_audios: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        return_tokens: bool = False,
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Optional[Union[torch.Tensor, List[torch.Tensor]]],
    ]:
        """Process audio with flexible input formats and optional speaker adaptation.

        Supports two modes:
        1. Standard codec (prompt_audios=None): Encode and decode audio
        2. Voice conversion (prompt_audios provided): Convert source to target speaker
           while preserving linguistic content (requires speaker_adapter)

        Args:
            source_audios: Audio input as single tensor (1, T) or list of
                tensors [(1, T1), (1, T2), ...] where T is time dimension.
            prompt_audios: Optional speaker reference audio matching source format.
                Single tensor (1, T) or list [(1, T1), (1, T2), ...].
                Used to extract target speaker identity for voice conversion.
            return_tokens: Whether to return ASR tokens (requires text_decoder).

        Returns:
            audios: Single tensor (1, T') or list [(1, T1'), ...] matching input format.
            tokens: Single tensor (L,) or list [(L1,), ...] if return_tokens is True, else None.

        Examples:
            >>> # Standard codec
            >>> reconstructed_audio, _ = bundle(audio)

            >>> # Voice conversion
            >>> converted_audio, _ = bundle(source_audio, prompt_audios=target_speaker_audio)

            >>> # Codec with ASR transcription
            >>> reconstructed_audio, transcript = bundle(audio, return_tokens=True)
        """
        is_list = isinstance(source_audios, list)

        inputs = self._pack_inputs(source_audios, prompt_audios)
        outputs = self._process(inputs, return_tokens=return_tokens)

        return self._unpack_outputs(outputs, as_list=is_list)
