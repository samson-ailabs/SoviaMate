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
        audio_inputs: torch.Tensor,
        audio_input_lengths: torch.Tensor,
        audio_target_lengths: torch.Tensor | None = None,
        speaker_prompts: torch.Tensor | None = None,
        speaker_prompt_lengths: torch.Tensor | None = None,
        apply_spec_augment: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Encode audio
        audio_encoder_outputs, audio_encoder_lengths = self.audio_encoder(
            audio_inputs, audio_input_lengths
        )

        # ASR decoder branch
        text_outputs = None
        text_output_lengths = None

        if self.text_decoder is not None:
            # Apply SpecAugment only for ASR branch
            text_encoder_features = audio_encoder_outputs
            text_encoder_lengths = audio_encoder_lengths

            if apply_spec_augment and self.spec_augment is not None:
                text_encoder_features = self.spec_augment(
                    text_encoder_features, text_encoder_lengths
                )

            text_outputs, text_output_lengths = self.text_decoder(
                text_encoder_features, text_encoder_lengths
            )

        # Quantize clean audio features (no augmentation)
        audio_quantized_outputs, audio_quantized_lengths = self.audio_quantizer(
            audio_encoder_outputs, audio_encoder_lengths
        )

        # Speaker adaptation after quantization
        if self.speaker_adapter is not None and speaker_prompts is not None:
            audio_quantized_outputs, audio_quantized_lengths = self.speaker_adapter(
                audio_quantized_outputs,
                audio_quantized_lengths,
                speaker_prompts,
                speaker_prompt_lengths,
            )

        # Decode to audio
        audio_outputs, audio_output_lengths = self.audio_decoder(
            audio_quantized_outputs, audio_quantized_lengths
        )

        if audio_target_lengths is not None:
            padding = audio_target_lengths.max() - audio_outputs.size(1)
            audio_outputs = F.pad(audio_outputs, (0, 0, 0, padding))
            audio_output_lengths = audio_target_lengths.clone()

        return (
            audio_outputs,
            audio_output_lengths,
            text_outputs,
            text_output_lengths,
        )

    def training_step(self, batch: Tuple[torch.Tensor, ...]):
        (
            audio_inputs,
            audio_input_lengths,
            speaker_prompts,
            speaker_prompt_lengths,
            audio_targets,
            audio_target_lengths,
            text_targets,
            text_target_lengths,
        ) = batch

        (audio_outputs, _, text_outputs, text_output_lengths) = self.forward(
            audio_inputs,
            audio_input_lengths,
            audio_target_lengths,
            speaker_prompts,
            speaker_prompt_lengths,
            apply_spec_augment=True,
        )

        # Transpose to channel-first format for discriminator
        audio_outputs = audio_outputs.transpose(1, 2)
        audio_targets = audio_targets.transpose(1, 2)

        disc_optim, gen_optim = self.optimizers()
        disc_sched, gen_sched = self.lr_schedulers()

        # Train discriminator
        self.toggle_optimizer(disc_optim)

        if random.random() < 0.5:  # Train discriminator every 50% of the time
            fake_segments, real_segments = self._get_random_segments(
                audio_outputs, audio_targets, audio_target_lengths
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

        audio_loss = self.audio_loss(audio_outputs, audio_targets, audio_target_lengths)

        fake_segments, _ = self._get_random_segments(
            audio_outputs, audio_targets, audio_target_lengths
        )

        fake_logits = self.discriminator(fake_segments)
        gen_loss = self.adv_loss(fake_logits)

        text_loss = 0.0
        if text_outputs is not None and self.text_loss is not None:
            text_loss = self.text_loss(
                logits=text_outputs,
                logit_lengths=text_output_lengths,
                targets=text_targets,
                target_lengths=text_target_lengths,
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
            audio_inputs,
            audio_input_lengths,
            speaker_prompts,
            speaker_prompt_lengths,
            audio_targets,
            audio_target_lengths,
            text_targets,
            text_target_lengths,
        ) = batch

        (audio_outputs, _, text_outputs, text_output_lengths) = self.forward(
            audio_inputs,
            audio_input_lengths,
            audio_target_lengths,
            speaker_prompts,
            speaker_prompt_lengths,
        )

        # Transpose to channel-first format for discriminator
        audio_outputs = audio_outputs.transpose(1, 2)
        audio_targets = audio_targets.transpose(1, 2)

        audio_loss = self.audio_loss(audio_outputs, audio_targets, audio_target_lengths)

        fake_segments, _ = self._get_random_segments(
            audio_outputs, audio_targets, audio_target_lengths
        )

        fake_logits = self.discriminator(fake_segments)
        gen_loss = self.adv_loss(fake_logits)

        text_loss = 0.0
        if text_outputs is not None and self.text_loss is not None:
            text_loss = self.text_loss(
                logits=text_outputs,
                logit_lengths=text_output_lengths,
                targets=text_targets,
                target_lengths=text_target_lengths,
            )

        val_loss = 2.0 * audio_loss + 1.0 * gen_loss + 0.5 * text_loss

        log_dict = {"val_audio_loss": audio_loss, "val_gen_loss": gen_loss}
        if text_loss > 0 and self.text_loss is not None:
            log_dict["val_text_loss"] = text_loss

        self.log_dict(log_dict, sync_dist=True)
        self.log("val_loss", val_loss, sync_dist=True, prog_bar=True)

    def _get_random_segments(
        self,
        decoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract random segments from decoder outputs and targets for discriminator.

        Args:
            decoder_outputs: Decoder outputs with shape (B, C, T).
            targets: Target audio with shape (B, C, T).
            target_lengths: Lengths of targets with shape (B,).

        Returns:
            Tuple of (decoder_segments, target_segments) with shape (B, C, SEGMENT_SIZE).
        """
        batch_size, num_channels, _ = decoder_outputs.size()
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

        decoder_segments = decoder_outputs[batch_indices, channel_indices, time_indices]
        target_segments = targets[batch_indices, channel_indices, time_indices]

        return decoder_segments, target_segments

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
