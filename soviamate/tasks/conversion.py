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

"""Audio Neural Codec Models for learning audio discrete tokens"""

import itertools
import os
from typing import Tuple

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from soviamate.modules.discriminator import SEGMENT_SIZE


class AudioCodecTask(L.LightningModule):
    """Audio Neural Codec Model for learning audio discrete tokens.

    Args:
        data (DictConfig): Configuration for the dataset.
        model (DictConfig): Configuration for the model.
        optim (DictConfig): Configuration for the optimizer and scheduler.
    """

    def __init__(self, data: DictConfig, model: DictConfig, optim: DictConfig) -> None:
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters("data", "model", "optim")

        self.audio_encoder = instantiate(model.audio_encoder)
        self.audio_quantizer = instantiate(model.audio_quantizer)
        self.audio_decoder = instantiate(model.audio_decoder)
        self.discriminator = instantiate(model.discriminator)

        self.audio_loss = instantiate(model.audio_loss)
        self.adv_loss = instantiate(model.adv_loss)

        # SpliceOut augmentation for ASR decoder
        if hasattr(model, "splice_out"):
            self.splice_out = instantiate(model.splice_out)
        else:
            self.splice_out = None

        # Text decoder for auxiliary training
        if hasattr(model, "text_decoder"):
            self.text_decoder = instantiate(model.text_decoder)
        else:
            self.text_decoder = None

        # Text loss for auxiliary training
        if hasattr(model, "text_loss"):
            self.text_loss = instantiate(model.text_loss)
        else:
            self.text_loss = None

        # Speaker adapter for speaker adaptation
        if hasattr(model, "speaker_adapter"):
            self.speaker_adapter = instantiate(model.speaker_adapter)
        else:
            self.speaker_adapter = None

        # Speaker contrastive loss for speaker disentanglement
        if hasattr(model, "speaker_loss"):
            self.speaker_loss = instantiate(model.speaker_loss)
        else:
            self.speaker_loss = None

        # Gradient scaling for speaker disentanglement (0.0=stop-grad, 1.0=normal)
        self.recon_grad_scale = getattr(model, "recon_grad_scale", 1.0)

    def train_dataloader(self) -> DataLoader:
        trainset = instantiate(
            self.hparams.data.trainset,
            _recursive_=False,
        )
        train_loader = DataLoader(
            dataset=trainset,
            shuffle=True,
            collate_fn=trainset.collate_data,
            **self.hparams.data.loader,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        valset = instantiate(
            self.hparams.data.valset,
            _recursive_=False,
        )
        val_loader = DataLoader(
            dataset=valset,
            shuffle=False,
            collate_fn=valset.collate_data,
            **self.hparams.data.loader,
        )
        return val_loader

    def forward(
        self,
        source_audios: torch.Tensor,
        source_audio_lengths: torch.Tensor,
        prompt_audios: torch.Tensor,
        prompt_audio_lengths: torch.Tensor,
        prompt_fbanks: torch.Tensor,
        prompt_fbank_lengths: torch.Tensor,
        target_audios: torch.Tensor,
        target_audio_lengths: torch.Tensor,
        apply_splice_out: bool = False,
    ) -> Tuple[torch.Tensor | None, ...]:
        # Extract speaker features from prompt waveform + clean fbank
        speaker_features, speaker_feature_lengths = None, None
        if self.speaker_adapter is not None:
            speaker_features, speaker_feature_lengths = self.speaker_adapter(
                prompt_audios, prompt_audio_lengths, prompt_fbanks, prompt_fbank_lengths
            )

        # Encode source (augmented) and optionally target (clean) audios
        if self.speaker_loss is not None:
            merged_audios = torch.cat([source_audios, target_audios], dim=0)
            merged_lengths = torch.cat(
                [source_audio_lengths, target_audio_lengths], dim=0
            )

            merged_features, merged_lengths = self.audio_encoder(
                merged_audios, merged_lengths
            )

            source_features, target_features = torch.chunk(
                merged_features, chunks=2, dim=0
            )
            source_feature_lengths, target_feature_lengths = torch.chunk(
                merged_lengths, chunks=2, dim=0
            )
        else:
            source_features, source_feature_lengths = self.audio_encoder(
                source_audios, source_audio_lengths
            )
            target_features, target_feature_lengths = None, None

        # Text recognition on source features (optional)
        output_tokens, output_token_lengths = None, None
        if self.text_decoder is not None:
            asr_features = source_features
            asr_feature_lengths = source_feature_lengths

            if apply_splice_out and self.splice_out is not None:
                asr_features, asr_feature_lengths = self.splice_out(
                    asr_features, asr_feature_lengths
                )

            output_tokens, output_token_lengths = self.text_decoder(
                asr_features, asr_feature_lengths
            )

        # Scale reconstruction gradients for speaker disentanglement
        if self.recon_grad_scale < 1.0:
            source_features = (
                source_features * self.recon_grad_scale
                + source_features.detach() * (1 - self.recon_grad_scale)
            )

        # Audio quantization
        quantized_outputs, quantized_lengths = self.audio_quantizer(
            source_features, source_feature_lengths
        )

        # Decode to audio with speaker conditioning
        output_audios, output_audio_lengths = self.audio_decoder(
            quantized_outputs,
            quantized_lengths,
            speaker_features,
            speaker_feature_lengths,
            target_audio_lengths.max().item(),
        )

        return (
            source_features,
            source_feature_lengths,
            target_features,
            target_feature_lengths,
            output_audios,
            output_audio_lengths,
            output_tokens,
            output_token_lengths,
        )

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        (
            source_audios,
            source_audio_lengths,
            prompt_audios,
            prompt_audio_lengths,
            prompt_fbanks,
            prompt_fbank_lengths,
            target_audios,
            target_audio_lengths,
            target_tokens,
            target_token_lengths,
        ) = batch

        (
            source_features,
            source_feature_lengths,
            target_features,
            target_feature_lengths,
            output_audios,
            _,
            output_tokens,
            output_token_lengths,
        ) = self.forward(
            source_audios,
            source_audio_lengths,
            prompt_audios,
            prompt_audio_lengths,
            prompt_fbanks,
            prompt_fbank_lengths,
            target_audios,
            target_audio_lengths,
            apply_splice_out=True,
        )

        output_audios = output_audios.transpose(1, 2)
        target_audios = target_audios.transpose(1, 2)

        disc_optim, gen_optim = self.optimizers()
        disc_sched, gen_sched = self.lr_schedulers()

        # Train discriminator
        self.toggle_optimizer(disc_optim)

        fake_segments, real_segments = self._get_random_segments(
            output_audios, target_audios, target_audio_lengths
        )

        fake_logits, real_logits = self.discriminator(
            fake_segments.detach(), real_segments.detach()
        )

        disc_loss = self.adv_loss(fake_logits, real_logits)
        self.manual_backward(disc_loss)

        disc_optim.step()
        disc_optim.zero_grad()
        disc_sched.step()

        self.untoggle_optimizer(disc_optim)

        # Train generator
        self.toggle_optimizer(gen_optim)

        audio_loss = self.audio_loss(output_audios, target_audios, target_audio_lengths)

        fake_segments, _ = self._get_random_segments(
            output_audios, target_audios, target_audio_lengths
        )

        fake_logits, _ = self.discriminator(fake_segments)
        gen_loss = self.adv_loss(fake_logits)

        text_loss = 0.0
        if output_tokens is not None and self.text_loss is not None:
            text_loss = self.text_loss(
                output_tokens,
                output_token_lengths,
                target_tokens,
                target_token_lengths,
            )

        speaker_loss = 0.0
        if self.speaker_loss is not None:
            speaker_loss = self.speaker_loss(
                source_features,
                target_features,
                source_feature_lengths,
                target_feature_lengths,
            )

        train_loss = (
            2.0 * audio_loss + 1.0 * gen_loss + 0.5 * text_loss + 0.5 * speaker_loss
        )
        self.manual_backward(train_loss)

        gen_optim.step()
        gen_optim.zero_grad()
        gen_sched.step()

        self.untoggle_optimizer(gen_optim)

        log_dict = {
            "train_audio_loss": audio_loss,
            "train_disc_loss": disc_loss,
            "train_gen_loss": gen_loss,
        }

        if text_loss > 0 and self.text_loss is not None:
            log_dict["train_text_loss"] = text_loss

        if speaker_loss > 0 and self.speaker_loss is not None:
            log_dict["train_speaker_loss"] = speaker_loss

        self.log_dict(log_dict, sync_dist=True)
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, ...]):
        (
            source_audios,
            source_audio_lengths,
            prompt_audios,
            prompt_audio_lengths,
            prompt_fbanks,
            prompt_fbank_lengths,
            target_audios,
            target_audio_lengths,
            target_tokens,
            target_token_lengths,
        ) = batch

        (
            source_features,
            source_feature_lengths,
            target_features,
            target_feature_lengths,
            output_audios,
            _,
            output_tokens,
            output_token_lengths,
        ) = self.forward(
            source_audios,
            source_audio_lengths,
            prompt_audios,
            prompt_audio_lengths,
            prompt_fbanks,
            prompt_fbank_lengths,
            target_audios,
            target_audio_lengths,
            apply_splice_out=False,
        )

        output_audios = output_audios.transpose(1, 2)
        target_audios = target_audios.transpose(1, 2)

        audio_loss = self.audio_loss(output_audios, target_audios, target_audio_lengths)

        fake_segments, _ = self._get_random_segments(
            output_audios, target_audios, target_audio_lengths
        )

        fake_logits, _ = self.discriminator(fake_segments)
        gen_loss = self.adv_loss(fake_logits)

        text_loss = 0.0
        if output_tokens is not None and self.text_loss is not None:
            text_loss = self.text_loss(
                output_tokens,
                output_token_lengths,
                target_tokens,
                target_token_lengths,
            )

        speaker_loss = 0.0
        if self.speaker_loss is not None:
            speaker_loss = self.speaker_loss(
                source_features,
                target_features,
                source_feature_lengths,
                target_feature_lengths,
            )

        val_loss = (
            2.0 * audio_loss + 1.0 * gen_loss + 0.5 * text_loss + 0.5 * speaker_loss
        )

        log_dict = {"val_audio_loss": audio_loss, "val_gen_loss": gen_loss}

        if text_loss > 0 and self.text_loss is not None:
            log_dict["val_text_loss"] = text_loss

        if speaker_loss > 0 and self.speaker_loss is not None:
            log_dict["val_speaker_loss"] = speaker_loss

        self.log_dict(log_dict, sync_dist=True)
        self.log("val_loss", val_loss, sync_dist=True, prog_bar=True)

    def _get_random_segments(
        self,
        output_audios: torch.Tensor,
        target_audios: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract random segments from output and target audios for discriminator.

        Args:
            output_audios: Model output audios with shape (B, C, T).
            target_audios: Target audios with shape (B, C, T).
            target_lengths: Lengths of targets with shape (B,).

        Returns:
            Tuple of (output_segments, target_segments) with shape (B, C, SEGMENT_SIZE).
        """
        batch_size, num_channels, _ = output_audios.size()
        max_start_index = target_lengths - SEGMENT_SIZE

        start_indices = torch.rand([batch_size], device=self.device)
        start_indices = (start_indices * max_start_index).long()

        batch_indices = torch.arange(batch_size, device=self.device)
        batch_indices = batch_indices.unsqueeze(1).unsqueeze(2)

        channel_indices = torch.arange(num_channels, device=self.device)
        channel_indices = channel_indices.unsqueeze(0).unsqueeze(2)

        time_offsets = torch.arange(SEGMENT_SIZE, device=self.device)
        time_offsets = time_offsets.unsqueeze(0).unsqueeze(0)

        time_indices = start_indices.unsqueeze(1).unsqueeze(2) + time_offsets

        output_segments = output_audios[batch_indices, channel_indices, time_indices]
        target_segments = target_audios[batch_indices, channel_indices, time_indices]

        return output_segments, target_segments

    def configure_optimizers(self):
        disc_optim = instantiate(
            self.hparams.optim.discriminator.optimizer,
            params=self.discriminator.parameters(),
        )
        disc_sched = instantiate(
            self.hparams.optim.discriminator.scheduler, optimizer=disc_optim
        )

        gen_param_groups = [
            self.audio_encoder.parameters(),
            self.audio_quantizer.parameters(),
            self.audio_decoder.parameters(),
        ]

        # Add text decoder parameters if available
        if self.text_decoder is not None:
            gen_param_groups.append(self.text_decoder.parameters())

        if self.text_loss is not None:
            gen_param_groups.append(self.text_loss.parameters())

        # Add speaker adaptation parameters if available
        if self.speaker_adapter is not None:
            gen_param_groups.append(self.speaker_adapter.parameters())

        if self.speaker_loss is not None:
            gen_param_groups.append(self.speaker_loss.parameters())

        gen_optim = instantiate(
            self.hparams.optim.generator.optimizer,
            params=itertools.chain(*gen_param_groups),
        )
        gen_sched = instantiate(
            self.hparams.optim.generator.scheduler, optimizer=gen_optim
        )

        return [disc_optim, gen_optim], [
            {"scheduler": disc_sched, "interval": "step"},
            {"scheduler": gen_sched, "interval": "step"},
        ]

    def export_model(self, filepath: str) -> None:
        """Export model for production using AudioCodecBundle.

        This method saves only the production-relevant components (encoder, quantizer,
        decoder, and optional text_decoder/speaker_adapter) along with hyperparameters.
        Training-only components (discriminator, losses, splice_out) are excluded.

        Args:
            filepath: Path where to save the checkpoint.
        """
        # Ensure directory exists
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)

        # Build model weights for main components
        model_weights = {
            "audio_encoder": self.audio_encoder.state_dict(),
            "audio_quantizer": self.audio_quantizer.state_dict(),
            "audio_decoder": self.audio_decoder.state_dict(),
        }

        # Optional components for ASR and speaker adaptation
        if self.text_decoder is not None:
            model_weights["text_decoder"] = self.text_decoder.state_dict()

        if self.speaker_adapter is not None:
            model_weights["speaker_adapter"] = self.speaker_adapter.state_dict()

        # Convert hyperparameters to plain dict for portable checkpoint
        hyper_parameters = {
            "data": OmegaConf.to_container(self.hparams.data, resolve=True),
            "model": OmegaConf.to_container(self.hparams.model, resolve=True),
        }

        # Save checkpoint
        checkpoint = {
            "model_weights": model_weights,
            "hyper_parameters": hyper_parameters,
        }
        torch.save(checkpoint, filepath)
        print(f"Exported model checkpoint to {filepath}")
