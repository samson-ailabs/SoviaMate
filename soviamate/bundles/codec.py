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

"""Audio codec bundle for production inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from soviamate.datas.tokenizer import SentencePieceTokenizer


@dataclass
class CodecInputs:
    """Structured inputs for AudioCodecBundle.

    Attributes:
        source_audios: Batched audio tensors (B, T, 1).
        source_audio_lengths: Length of each audio (B,).
        prompt_audios: Optional speaker prompts (B, T_prompt, 1).
        prompt_audio_lengths: Optional prompt lengths (B,).
    """

    source_audios: torch.Tensor
    source_audio_lengths: torch.Tensor
    prompt_audios: Optional[torch.Tensor] = None
    prompt_audio_lengths: Optional[torch.Tensor] = None


@dataclass
class CodecOutputs:
    """Structured outputs from AudioCodecBundle.

    Attributes:
        audios: Decoded audio, shape ``(B, T_max, 1)``.
        audio_lengths: Per-sample lengths, shape ``(B,)``.
        tokens: Optional greedy-argmax ASR token ids, shape ``(B, T_tok)``.
        token_lengths: Optional per-sample token lengths, shape ``(B,)``.
    """

    audios: torch.Tensor
    audio_lengths: torch.Tensor
    tokens: Optional[torch.Tensor] = None
    token_lengths: Optional[torch.Tensor] = None


@dataclass
class CodecStreamState:
    """Per-session state for streaming inference through AudioCodecBundle.

    Attributes:
        chunk_size: Feature frames per chunk.
        speaker_embeddings: Optional speaker embedding for voice conversion.
        encoder_caches: Encoder layer caches; ``None`` on a cold start.
        decoder_caches: Decoder layer caches; ``None`` on a cold start.
        return_text: Whether to decode a transcript for each chunk.
        text_caches: Text-decoder layer caches; ``None`` on a cold start.
        last_token: Last emitted non-blank token for cross-chunk CTC
            deduplication; ``-1`` on a cold start.
    """

    chunk_size: int
    speaker_embeddings: Optional[torch.Tensor] = None
    encoder_caches: Optional[List[List[torch.Tensor]]] = None
    decoder_caches: Optional[List[List[torch.Tensor]]] = None
    return_text: bool = False
    text_caches: Optional[List[List[torch.Tensor]]] = None
    last_token: int = -1


class AudioCodecBundle(nn.Module):
    """Audio codec bundle for production inference.

    Bundles the audio encoder, quantizer, and decoder with optional ASR text
    decoder and speaker adapter. Exposes two inference modes:

    * :meth:`forward` — full-context reconstruction or voice conversion.
    * :meth:`init_stream` + :meth:`stream_chunk` — chunked streaming
      inference with persistent encoder/decoder/text caches.

    Load via :meth:`from_checkpoint` on a file produced by
    ``AudioCodecTask.export_model()``.

    Args:
        audio_encoder: Audio encoder module.
        audio_quantizer: Audio quantizer module.
        audio_decoder: Audio decoder module with speaker conditioning.
        text_decoder: Optional ASR text decoder module.
        speaker_adapter: Optional frozen speaker encoder for voice conversion.
        tokenizer: Optional tokenizer for decoding token ids to text.
        device: Device to place the model on (``'cpu'``, ``'cuda'``, ...).
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        audio_quantizer: nn.Module,
        audio_decoder: nn.Module,
        text_decoder: Optional[nn.Module] = None,
        speaker_adapter: Optional[nn.Module] = None,
        tokenizer: Optional[SentencePieceTokenizer] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        self.n_mels = 80  # Must match speaker adapter config
        self.sample_rate = 16000  # Must match speaker adapter config

        self.audio_encoder = audio_encoder
        self.audio_quantizer = audio_quantizer
        self.audio_decoder = audio_decoder
        self.text_decoder = text_decoder
        self.tokenizer = tokenizer
        self.speaker_adapter = speaker_adapter

        if device is not None:
            self.to(device)

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

        # Speaker adapter for inference only (frozen, not trained)
        speaker_adapter = None
        if "speaker_adapter" in model_config:
            speaker_adapter = instantiate(model_config["speaker_adapter"])

        # Load model weights
        audio_encoder.load_state_dict(audio_encoder_weights)
        audio_quantizer.load_state_dict(audio_quantizer_weights)
        audio_decoder.load_state_dict(audio_decoder_weights)

        if text_decoder is not None and text_decoder_weights:
            text_decoder.load_state_dict(text_decoder_weights)

        # Tokenizer for text decoding
        tokenizer = None
        data_config = hparams.get("data", {})
        if "tokenizer" in data_config and text_decoder is not None:
            tokenizer = instantiate(data_config["tokenizer"])

        # Build bundle
        bundle = cls(
            audio_encoder=audio_encoder,
            audio_quantizer=audio_quantizer,
            audio_decoder=audio_decoder,
            text_decoder=text_decoder,
            speaker_adapter=speaker_adapter,
            tokenizer=tokenizer,
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

        batch_size = source_audios.size(0)
        prompt_audio_lengths = None

        if prompt_audios is not None:
            if isinstance(prompt_audios, list):
                if len(prompt_audios) != batch_size:
                    raise ValueError(
                        f"Number of prompts ({len(prompt_audios)}) must match "
                        f"number of audios ({batch_size})"
                    )
                prompt_audios, prompt_audio_lengths = self.pack_sequence(prompt_audios)
            else:
                prompt_audios = (
                    prompt_audios.t().unsqueeze(0).expand(batch_size, -1, -1)
                )
                prompt_audio_lengths = torch.full(
                    (batch_size,),
                    prompt_audios.size(1),
                    dtype=torch.long,
                    device=prompt_audios.device,
                )

        return CodecInputs(
            source_audios=source_audios,
            source_audio_lengths=source_audio_lengths,
            prompt_audios=prompt_audios,
            prompt_audio_lengths=prompt_audio_lengths,
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

    def _ctc_collapse(
        self, tokens: torch.Tensor, blank: int = 0, last_token: int = -1
    ) -> Tuple[str, int]:
        """Collapse CTC tokens, decode to text, and report the new last token.

        Removes consecutive duplicates and blanks. When ``last_token >= 0``
        (streaming mode), prepends it to both the collapse (for cross-chunk
        dedup) and the tokenizer decode (so SentencePiece preserves the ``▁``
        word-boundary prefix at the chunk start), then strips the prefix
        contribution so consecutive chunk transcripts concatenate cleanly.

        Args:
            tokens (Tensor): Raw per-frame token ids ``(T,)``.
            blank (int): Blank token index. Default: ``0``.
            last_token (int): Last emitted token from the previous chunk;
                ``-1`` for full-context decoding (no cross-chunk context).

        Returns:
            Tuple[str, int]: ``(text, new_last_token)``. ``new_last_token``
                is ``collapsed[-1]`` if this chunk emitted any tokens,
                otherwise the input ``last_token`` unchanged.
        """
        if last_token >= 0:
            prefix = torch.tensor([last_token], device=tokens.device)
            tokens = torch.cat([prefix, tokens])

        unique = torch.unique_consecutive(tokens)
        collapsed = unique[(unique != blank) & (unique != -1)].tolist()

        if last_token >= 0 and collapsed and collapsed[0] == last_token:
            collapsed = collapsed[1:]

        if not collapsed:
            return "", last_token

        if last_token >= 0:
            ctx = [last_token]
            ctx_len = len(self.tokenizer.decode(ctx))
            text = self.tokenizer.decode(ctx + collapsed)[ctx_len:]
        else:
            text = self.tokenizer.decode(collapsed)

        return text, collapsed[-1]

    def _process(
        self, codec_inputs: CodecInputs, return_text: bool = False
    ) -> CodecOutputs:
        """Process audio through the codec pipeline with optional speaker conditioning.

        Args:
            codec_inputs: Structured inputs with batched tensors.
            return_text: Whether to run the text decoder and return a transcript.

        Returns:
            CodecOutputs with processed audio and optional tokens.
        """
        # Validate optional components
        if return_text and self.text_decoder is None:
            raise ValueError(
                "Cannot return transcript: text_decoder not available in bundle."
            )

        if codec_inputs.prompt_audios is not None and self.speaker_adapter is None:
            raise ValueError(
                "Cannot use prompt audio: speaker_adapter not available in bundle."
            )

        # Encode source audio
        source_features, source_feature_lengths = self.audio_encoder(
            codec_inputs.source_audios, codec_inputs.source_audio_lengths
        )

        # Text decoder forward (optional)
        output_tokens = None
        token_lengths = None

        if return_text:
            logits, token_lengths = self.text_decoder(
                source_features, source_feature_lengths
            )
            output_tokens = logits.argmax(dim=-1)

        # Quantize
        quantized_outputs, quantized_lengths = self.audio_quantizer(
            source_features, source_feature_lengths
        )

        # Extract speaker embedding from prompt (optional)
        speaker_embeddings = None

        if self.speaker_adapter is not None and codec_inputs.prompt_audios is not None:
            prompt_fbanks, prompt_fbank_lengths = self._compute_fbank(
                codec_inputs.prompt_audios, codec_inputs.prompt_audio_lengths
            )
            speaker_embeddings = self.speaker_adapter(
                prompt_fbanks, prompt_fbank_lengths
            )

        # Decode with speaker conditioning; clamp output to source length so
        # the padded-batch output matches per-sample audio lengths exactly.
        max_output_length = codec_inputs.source_audio_lengths.max().item()
        output_audios, output_lengths = self.audio_decoder(
            quantized_outputs,
            quantized_lengths,
            speaker_embeddings=speaker_embeddings,
            max_output_length=max_output_length,
        )

        return CodecOutputs(
            audios=output_audios,
            audio_lengths=output_lengths,
            tokens=output_tokens,
            token_lengths=token_lengths,
        )

    def _unpack_outputs(
        self, codec_outputs: CodecOutputs, as_list: bool = False
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]], Optional[Union[str, List[str]]]
    ]:
        """Unpack outputs to requested format.

        Args:
            codec_outputs: Batched outputs from process().
            as_list: Whether to return as list.

        Returns:
            Tuple of (audios, texts). If as_list is True, returns lists.
            If False, returns first element.
        """
        audio_list = self.unpack_sequence(
            codec_outputs.audios, codec_outputs.audio_lengths
        )

        text_list = None
        if codec_outputs.tokens is not None:
            text_list = []
            for i in range(codec_outputs.tokens.size(0)):
                text, _ = self._ctc_collapse(
                    codec_outputs.tokens[i, : codec_outputs.token_lengths[i]]
                )
                text_list.append(text)

        if as_list:
            return audio_list, text_list

        return audio_list[0], text_list[0] if text_list else None

    @torch.inference_mode()
    def forward(
        self,
        source_audios: Union[torch.Tensor, List[torch.Tensor]],
        prompt_audios: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        return_text: bool = False,
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Optional[Union[str, List[str]]],
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
            return_text: Whether to run the text decoder and return a transcript
                (requires ``text_decoder`` and ``tokenizer``).

        Returns:
            audios: Single tensor ``(1, T')`` or list ``[(1, T1'), ...]`` matching
                the input format.
            texts: Decoded transcript (single string or list of strings) when
                ``return_text`` is True; ``None`` otherwise.

        Examples:
            >>> # Standard codec
            >>> reconstructed_audio, _ = bundle(audio)

            >>> # Voice conversion
            >>> converted_audio, _ = bundle(source_audio, prompt_audios=target_speaker_audio)

            >>> # Codec with ASR transcription
            >>> reconstructed_audio, transcript = bundle(audio, return_text=True)
        """
        is_list = isinstance(source_audios, list)

        inputs = self._pack_inputs(source_audios, prompt_audios)
        outputs = self._process(inputs, return_text=return_text)

        return self._unpack_outputs(outputs, as_list=is_list)

    def init_stream(
        self,
        chunk_size: int,
        prompt_audio: Optional[torch.Tensor] = None,
        return_text: bool = False,
    ) -> CodecStreamState:
        """Initialize a streaming inference session.

        Configures encoder/decoder streaming chunk size on the bundle's
        modules and precomputes the speaker embedding if a prompt is given.
        Returns a per-session state object; pass it to :meth:`stream_chunk`
        for each incoming chunk.

        Args:
            chunk_size: Feature frames per chunk. Each audio chunk passed
                to :meth:`stream_chunk` must be exactly
                ``chunk_size · frame_stacking · hop_length`` samples.
            prompt_audio: Optional speaker prompt ``(1, T)`` for voice
                conversion. Its embedding is computed once here and reused
                across all chunks for this session.
            return_text: Whether to run the text decoder on each chunk
                and return an incremental transcript. Requires both
                ``text_decoder`` and ``tokenizer`` in the bundle.

        Returns:
            Per-session state to thread through :meth:`stream_chunk`.
        """
        if return_text and (self.text_decoder is None or self.tokenizer is None):
            raise ValueError(
                "Cannot return text: text_decoder or tokenizer not available in bundle."
            )

        self.audio_encoder.set_streaming_config(chunk_size)
        self.audio_decoder.set_streaming_config(chunk_size)

        if return_text:
            self.text_decoder.set_streaming_config(chunk_size)

        speaker_embeddings = None
        if prompt_audio is not None:
            if self.speaker_adapter is None:
                raise ValueError(
                    "Cannot use prompt_audio: speaker_adapter not available in bundle."
                )

            prompt = prompt_audio.t().unsqueeze(0).to(self.device)
            length = torch.tensor(
                [prompt.size(1)], dtype=torch.long, device=self.device
            )

            fbanks, fbank_lengths = self._compute_fbank(prompt, length)
            speaker_embeddings = self.speaker_adapter(fbanks, fbank_lengths)

        return CodecStreamState(
            chunk_size=chunk_size,
            speaker_embeddings=speaker_embeddings,
            return_text=return_text,
        )

    @torch.inference_mode()
    def stream_chunk(
        self, audio_chunk: torch.Tensor, state: CodecStreamState
    ) -> Tuple[torch.Tensor, Optional[str], CodecStreamState]:
        """Process one chunk of audio through a streaming session.

        Args:
            audio_chunk: Shape ``(1, chunk_samples)`` where
                ``chunk_samples = chunk_size · frame_stacking · hop_length``.
            state: Session state from :meth:`init_stream`. Mutated in place
                with updated caches and returned for convenience.

        Returns:
            waveform_chunk: Decoded waveform for the current chunk only, shape
                ``(1, chunk_samples)``. Not cumulative — concatenate across
                calls if you want full-session audio.
            text_chunk: Incremental transcript emitted for the current chunk,
                CTC-collapsed and deduplicated against the previous chunk's
                last emission. ``None`` when ``return_text`` was not set.
            state: Same state instance, with caches advanced to the next call.

        Example:
            >>> state = bundle.init_stream(chunk_size=8, prompt_audio=prompt)
            >>> for chunk in audio_chunks:
            ...     wav_chunk, _, state = bundle.stream_chunk(chunk, state)
        """
        audio_chunk = audio_chunk.t().unsqueeze(0).to(self.device)
        features, state.encoder_caches = self.audio_encoder.infer(
            audio_chunk, state.encoder_caches
        )

        text_chunk = None
        if state.return_text:
            logits, state.text_caches = self.text_decoder.infer(
                features, state.text_caches
            )
            text_chunk, state.last_token = self._ctc_collapse(
                logits.argmax(dim=-1)[0], last_token=state.last_token
            )

        quantized, _ = self.audio_quantizer(features)
        audio_chunk, state.decoder_caches = self.audio_decoder.infer(
            quantized, state.decoder_caches, state.speaker_embeddings
        )

        return audio_chunk.squeeze(-1), text_chunk, state
