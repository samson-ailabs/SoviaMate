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
import random
from typing import Tuple

import lightning as L
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from soviamate.modules.discriminator import SEGMENT_SIZE


class AudioCodecTask(L.LightningModule):
    """Audio Neural Codec Model for learning audio discrete tokens.

    Args:
        data (DictConfig): Configuration for the dataset.
        model (DictConfig): Configuration for the model.
        optim (DictConfig): Configuration for the optimizer and scheduler.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters("data", "model", "optim")

        self.audio_encoder = instantiate(self.hparams.model.audio_encoder)
        self.audio_quantizer = instantiate(self.hparams.model.audio_quantizer)
        self.audio_decoder = instantiate(self.hparams.model.audio_decoder)

        self.discriminator = instantiate(self.hparams.model.discriminator)

        self.audio_loss = instantiate(self.hparams.model.audio_loss)
        self.adv_loss = instantiate(self.hparams.model.adv_loss)

        # SpecAugment for hidden representations
        if hasattr(self.hparams.model, "spec_augment"):
            self.spec_augment = instantiate(self.hparams.model.spec_augment)
        else:
            self.spec_augment = None

        # ASR decoder for auxiliary training
        if hasattr(self.hparams.model, "text_decoder"):
            self.text_decoder = instantiate(self.hparams.model.text_decoder)
        else:
            self.text_decoder = None

        # ASR loss for auxiliary training
        if hasattr(self.hparams.model, "text_loss"):
            self.text_loss = instantiate(self.hparams.model.text_loss)
        else:
            self.text_loss = None

        # Speaker adapter for speaker adaptation
        if hasattr(self.hparams.model, "speaker_adapter"):
            self.speaker_adapter = instantiate(self.hparams.model.speaker_adapter)
        else:
            self.speaker_adapter = None

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
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor | None = None,
        prompt_audios: torch.Tensor | None = None,
        prompt_lengths: torch.Tensor | None = None,
        apply_spec_augment: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Encode audio (shared encoder for both audio codec and ASR)
        encoder_features, encoder_lengths = self.audio_encoder(
            source_audios, source_lengths
        )

        # ASR branch (text recognition)
        output_tokens = None
        output_token_lengths = None

        if self.text_decoder is not None:
            asr_features = (
                self.spec_augment(encoder_features, encoder_lengths)
                if apply_spec_augment and self.spec_augment is not None
                else encoder_features
            )

            output_tokens, output_token_lengths = self.text_decoder(
                asr_features, encoder_lengths
            )

        # Audio codec branch (restoration)
        quantized_features, quantized_lengths = self.audio_quantizer(
            encoder_features, encoder_lengths
        )

        # Speaker adaptation
        if self.speaker_adapter is not None and prompt_audios is not None:
            quantized_features, quantized_lengths = self.speaker_adapter(
                quantized_features,
                quantized_lengths,
                prompt_audios,
                prompt_lengths,
            )

        # Decode to audio
        output_audios, output_lengths = self.audio_decoder(
            quantized_features, quantized_lengths
        )

        if target_lengths is not None:
            padding = target_lengths.max() - output_audios.size(1)
            output_audios = F.pad(output_audios, (0, 0, 0, padding))
            output_lengths = target_lengths.clone()

        return output_audios, output_lengths, output_tokens, output_token_lengths

    def training_step(self, batch: Tuple[torch.Tensor, ...]):
        (
            source_audios,
            source_lengths,
            prompt_audios,
            prompt_lengths,
            target_audios,
            target_lengths,
            target_tokens,
            target_token_lengths,
        ) = batch

        output_audios, _, output_tokens, output_token_lengths = self.forward(
            source_audios,
            source_lengths,
            target_lengths,
            prompt_audios,
            prompt_lengths,
            apply_spec_augment=True,
        )

        # Transpose to channel-first format for discriminator
        output_audios = output_audios.transpose(1, 2)
        target_audios = target_audios.transpose(1, 2)

        disc_optim, gen_optim = self.optimizers()
        disc_sched, gen_sched = self.lr_schedulers()

        # Train discriminator
        self.toggle_optimizer(disc_optim)

        if self.global_step == 0 or random.random() < 0.5:
            fake_segments, real_segments = self._get_random_segments(
                output_audios, target_audios, target_lengths
            )

            fake_logits = self.discriminator(fake_segments.detach())
            real_logits = self.discriminator(real_segments.detach())

            disc_loss = self.adv_loss(fake_logits, real_logits)
            self.manual_backward(disc_loss)

            disc_optim.step()
            disc_optim.zero_grad()

        disc_sched.step()

        self.untoggle_optimizer(disc_optim)

        # Train generator
        self.toggle_optimizer(gen_optim)

        audio_loss = self.audio_loss(output_audios, target_audios, target_lengths)

        fake_segments, _ = self._get_random_segments(
            output_audios, target_audios, target_lengths
        )

        fake_logits = self.discriminator(fake_segments)
        gen_loss = self.adv_loss(fake_logits)

        text_loss = 0.0
        if output_tokens is not None and self.text_loss is not None:
            text_loss = self.text_loss(
                logits=output_tokens,
                logit_lengths=output_token_lengths,
                targets=target_tokens,
                target_lengths=target_token_lengths,
            )

        train_loss = 2.0 * audio_loss + 1.0 * gen_loss + 0.5 * text_loss
        self.manual_backward(train_loss)

        gen_optim.step()
        gen_optim.zero_grad()
        gen_sched.step()

        self.untoggle_optimizer(gen_optim)

        # Log losses
        log_dict = {"train_audio_loss": audio_loss, "train_gen_loss": gen_loss}
        if "disc_loss" in locals():
            log_dict["train_disc_loss"] = disc_loss
        if text_loss > 0 and self.text_loss is not None:
            log_dict["train_text_loss"] = text_loss

        self.log_dict(log_dict, sync_dist=True)
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, ...]):
        (
            source_audios,
            source_lengths,
            prompt_audios,
            prompt_lengths,
            target_audios,
            target_lengths,
            target_tokens,
            target_token_lengths,
        ) = batch

        output_audios, _, output_tokens, output_token_lengths = self.forward(
            source_audios,
            source_lengths,
            target_lengths,
            prompt_audios,
            prompt_lengths,
        )

        # Transpose to channel-first format for discriminator
        output_audios = output_audios.transpose(1, 2)
        target_audios = target_audios.transpose(1, 2)

        audio_loss = self.audio_loss(output_audios, target_audios, target_lengths)

        fake_segments, _ = self._get_random_segments(
            output_audios, target_audios, target_lengths
        )

        fake_logits = self.discriminator(fake_segments)
        gen_loss = self.adv_loss(fake_logits)

        text_loss = 0.0
        if output_tokens is not None and self.text_loss is not None:
            text_loss = self.text_loss(
                logits=output_tokens,
                logit_lengths=output_token_lengths,
                targets=target_tokens,
                target_lengths=target_token_lengths,
            )

        val_loss = 2.0 * audio_loss + 1.0 * gen_loss + 0.5 * text_loss

        log_dict = {"val_audio_loss": audio_loss, "val_gen_loss": gen_loss}
        if text_loss > 0 and self.text_loss is not None:
            log_dict["val_text_loss"] = text_loss

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
            self.hparams.optim.discriminator.scheduler,
            optimizer=disc_optim,
        )

        gen_param_groups = [
            self.audio_encoder.parameters(),
            self.audio_quantizer.parameters(),
            self.audio_decoder.parameters(),
        ]

        # Add ASR parameters if available
        if self.text_decoder is not None:
            gen_param_groups.append(self.text_decoder.parameters())

        # Add speaker adaptation parameters if available
        if self.speaker_adapter is not None:
            gen_param_groups.append(self.speaker_adapter.parameters())

        gen_optim = instantiate(
            self.hparams.optim.generator.optimizer,
            params=itertools.chain(*gen_param_groups),
        )

        # Same handling for generator scheduler
        gen_sched = instantiate(
            self.hparams.optim.generator.scheduler,
            optimizer=gen_optim,
        )

        return [disc_optim, gen_optim], [
            {"scheduler": disc_sched, "interval": "step"},
            {"scheduler": gen_sched, "interval": "step"},
        ]

    def export_model(self, filepath: str) -> None:
        """Export model for production using AudioCodecBundle.

        This method saves only the production-relevant components (encoder, quantizer,
        decoder, and optional text_decoder/speaker_adapter) along with hyperparameters.
        Training-only components (discriminator, losses, spec_augment) are excluded.

        Args:
            filepath: Path where to save the checkpoint.
        """
        # Build state dict with nested structure
        state_dict = {
            "audio_encoder": self.audio_encoder.state_dict(),
            "audio_quantizer": self.audio_quantizer.state_dict(),
            "audio_decoder": self.audio_decoder.state_dict(),
        }

        # Optional components
        if self.text_decoder is not None:
            state_dict["text_decoder"] = self.text_decoder.state_dict()

        if self.speaker_adapter is not None:
            state_dict["speaker_adapter"] = self.speaker_adapter.state_dict()

        # Create checkpoint with model components and hyperparameters
        hparams = OmegaConf.to_container(self.hparams.model, resolve=True)
        checkpoint = {"state_dict": state_dict, "hyper_parameters": hparams}

        # Save checkpoint
        torch.save(checkpoint, filepath)
        print(f"Exported model checkpoint to: {filepath}")
