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
from typing import Optional, Tuple

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

        # ASR components
        if hasattr(self.hparams.model, "text_decoder"):
            self.text_decoder = instantiate(self.hparams.model.text_decoder)
        else:
            self.text_decoder = None

        # Speaker adaptation components
        if hasattr(self.hparams.model, "speaker_encoder"):
            self.speaker_encoder = instantiate(self.hparams.model.speaker_encoder)
        else:
            self.speaker_encoder = None

        if hasattr(self.hparams.model, "speaker_adapter"):
            self.speaker_adapter = instantiate(self.hparams.model.speaker_adapter)
        else:
            self.speaker_adapter = None

        self.audio_quantizer = instantiate(self.hparams.model.audio_quantizer)
        self.audio_decoder = instantiate(self.hparams.model.audio_decoder)

        self.discriminator = instantiate(self.hparams.model.discriminator)

        self.audio_loss = instantiate(self.hparams.model.audio_loss)
        self.adv_loss = instantiate(self.hparams.model.adv_loss)

        # ASR loss if text decoder is present
        if self.text_decoder is not None:
            if hasattr(self.hparams.model, "text_loss"):
                self.text_loss = instantiate(self.hparams.model.text_loss)
            else:
                self.text_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def train_dataloader(self) -> DataLoader:
        trainset = instantiate(
            self.hparams.data.trainset,
            _recursive_=False,
        )
        train_dl = DataLoader(
            dataset=trainset,
            shuffle=True,
            collate_fn=trainset.collate_data,
            **self.hparams.data.loader,
        )
        return train_dl

    def val_dataloader(self):
        valset = instantiate(
            self.hparams.data.valset,
            _recursive_=False,
        )
        val_dl = DataLoader(
            dataset=valset,
            shuffle=False,
            collate_fn=valset.collate_data,
            **self.hparams.data.loader,
        )
        return val_dl

    def forward(
        self,
        input_audios: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        speaker_prompts: Optional[torch.Tensor] = None,
        speaker_prompt_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode audio
        encoder_outputs, encoder_lengths = self.audio_encoder(
            input_audios, input_lengths
        )

        # ASR: Extract text before quantization
        asr_outputs = None
        if self.text_decoder is not None:
            asr_outputs = self.text_decoder(encoder_outputs)

        # Quantize the features first
        quantized_outputs = self.audio_quantizer(encoder_outputs)

        # Speaker adaptation AFTER quantization
        if self.speaker_encoder is not None and speaker_prompts is not None:
            # Extract speaker embeddings from prompt
            spk_utt_embs, spk_frm_embs, spk_frm_lens = self.speaker_encoder(
                speaker_prompts, speaker_prompt_lengths
            )

            # Apply speaker adaptation on quantized features
            if self.speaker_adapter is not None:
                quantized_outputs, encoder_lengths = self.speaker_adapter(
                    quantized_outputs,
                    encoder_lengths,
                    spk_utt_embs,
                    spk_frm_embs,
                    spk_frm_lens,
                )

        # Decode to audio
        decoder_outputs, decoder_lengths = self.audio_decoder(
            quantized_outputs, encoder_lengths
        )

        if target_lengths is not None:
            decoder_outputs = F.pad(
                decoder_outputs,
                (0, 0, 0, target_lengths.max() - decoder_outputs.size(1)),
            )
            decoder_lengths = target_lengths.clone()

        return decoder_outputs, decoder_lengths, asr_outputs, encoder_outputs

    def training_step(self, batch: Tuple[torch.Tensor, ...]):
        (
            input_audios,
            input_lengths,
            speaker_prompts,
            speaker_prompt_lengths,
            target_audios,
            target_lengths,
            target_texts,
            target_text_lengths,
        ) = batch

        # Forward pass with speaker adaptation
        forward_outputs = self.forward(
            input_audios,
            input_lengths,
            target_lengths,
            speaker_prompts,
            speaker_prompt_lengths,
        )

        output_audios, _, asr_outputs, encoder_features = forward_outputs

        output_audios = output_audios.transpose(1, 2)
        target_audios = target_audios.transpose(1, 2)

        disc_optim, gen_optim = self.optimizers()
        disc_sched, gen_sched = self.lr_schedulers()

        # Train discriminator
        self.toggle_optimizer(disc_optim)

        if random.random() < 0.5:  # Train discriminator every 50% of the time
            outputs, targets = self._get_random_segments(
                output_audios, target_audios, target_lengths
            )

            fake_logits = self.discriminator(outputs.detach())
            real_logits = self.discriminator(targets.detach())

            disc_loss = self.adv_loss(fake_logits, real_logits)
            self.manual_backward(disc_loss)

            disc_optim.step()
            disc_optim.zero_grad()

        disc_sched.step()  # Update discriminator scheduler every step

        self.untoggle_optimizer(disc_optim)

        # Train Generator
        self.toggle_optimizer(gen_optim)

        audio_loss = self.audio_loss(output_audios, target_audios, target_lengths)

        outputs, _ = self._get_random_segments(
            output_audios, target_audios, target_lengths
        )

        logits = self.discriminator(outputs)
        gen_loss = self.adv_loss(logits)

        # Add ASR loss if available (sequence-to-sequence loss)
        text_loss = 0.0
        if (
            asr_outputs is not None
            and target_texts is not None
            and encoder_features is not None
        ):
            # Use sequence-to-sequence loss (CTC + RNN-T)
            text_loss = self.text_loss(
                encoder_outputs=encoder_features,  # Use pre-quantized features
                encoder_lengths=input_lengths,
                decoder_outputs=asr_outputs,
                decoder_lengths=input_lengths,  # Use same lengths as encoder
                target_tokens=target_texts,
                target_lengths=target_text_lengths,
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
        if asr_outputs is not None:
            log_dict["train_text_loss"] = text_loss

        self.log_dict(log_dict, sync_dist=True)
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, ...]):
        (
            input_audios,
            input_lengths,
            speaker_prompts,
            speaker_prompt_lengths,
            target_audios,
            target_lengths,
            target_texts,
            target_text_lengths,
        ) = batch

        # Forward pass
        forward_outputs = self.forward(
            input_audios,
            input_lengths,
            target_lengths,
            speaker_prompts,
            speaker_prompt_lengths,
        )

        output_audios, _, asr_outputs, encoder_features = forward_outputs

        output_audios = output_audios.transpose(1, 2)
        target_audios = target_audios.transpose(1, 2)

        audio_loss = self.audio_loss(output_audios, target_audios, target_lengths)

        outputs, _ = self._get_random_segments(
            output_audios, target_audios, target_lengths
        )

        logits = self.discriminator(outputs)
        gen_loss = self.adv_loss(logits)

        # Add ASR loss if available (sequence-to-sequence loss)
        text_loss = 0.0
        if (
            asr_outputs is not None
            and target_texts is not None
            and encoder_features is not None
        ):
            # Use sequence-to-sequence loss (CTC + RNN-T)
            text_loss = self.text_loss(
                encoder_outputs=encoder_features,  # Use pre-quantized features
                encoder_lengths=input_lengths,
                decoder_outputs=asr_outputs,
                decoder_lengths=input_lengths,  # Use same lengths as encoder
                target_tokens=target_texts,
                target_lengths=target_text_lengths,
            )

        val_loss = 2.0 * audio_loss + 1.0 * gen_loss + 0.5 * text_loss

        log_dict = {"val_audio_loss": audio_loss, "val_gen_loss": gen_loss}
        if asr_outputs is not None:
            log_dict["val_text_loss"] = text_loss

        self.log_dict(log_dict, sync_dist=True)
        self.log("val_loss", val_loss, sync_dist=True, prog_bar=True)

    def _get_random_segments(
        self, outputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, _ = outputs.size()
        max_start_index = lengths - SEGMENT_SIZE

        start_indices = torch.rand([b], device=self.device)
        start_indices = (start_indices * max_start_index).long()

        batch_indices = torch.arange(b, device=self.device)
        batch_indices = batch_indices.unsqueeze(1).unsqueeze(2)

        channel_indices = torch.arange(c, device=self.device)
        channel_indices = channel_indices.unsqueeze(0).unsqueeze(2)

        time_offsets = torch.arange(SEGMENT_SIZE, device=self.device)
        time_offsets = time_offsets.unsqueeze(0).unsqueeze(0)

        time_indices = start_indices.unsqueeze(1).unsqueeze(2) + time_offsets

        clipped_outputs = outputs[batch_indices, channel_indices, time_indices]
        clipped_targets = targets[batch_indices, channel_indices, time_indices]

        return clipped_outputs, clipped_targets

    def configure_optimizers(self):
        disc_params = self.discriminator.parameters()

        disc_optim = instantiate(
            self.hparams.optim.discriminator.optimizer, params=disc_params
        )

        # For SequentialLR, we need to create the nested schedulers manually
        # since Hydra can't instantiate them with the optimizer dependency
        scheduler_config = self.hparams.optim.discriminator.scheduler

        if hasattr(scheduler_config, "schedulers"):
            # Create the nested schedulers first
            nested_schedulers = []
            for sched_cfg in scheduler_config.schedulers:
                nested_scheduler = instantiate(sched_cfg, optimizer=disc_optim)
                nested_schedulers.append(nested_scheduler)

            # Create the SequentialLR with the nested schedulers
            disc_sched = instantiate(
                scheduler_config, optimizer=disc_optim, schedulers=nested_schedulers
            )
        else:
            # Regular scheduler instantiation
            disc_sched = instantiate(scheduler_config, optimizer=disc_optim)

        gen_param_groups = [
            self.audio_encoder.parameters(),
            self.audio_quantizer.parameters(),
            self.audio_decoder.parameters(),
        ]

        # Add ASR parameters if available
        if self.text_decoder is not None:
            gen_param_groups.append(self.text_decoder.parameters())

        # Add speaker adaptation parameters if available
        if self.speaker_encoder is not None:
            gen_param_groups.append(self.speaker_encoder.parameters())

        if self.speaker_adapter is not None:
            gen_param_groups.append(self.speaker_adapter.parameters())

        gen_params = itertools.chain(*gen_param_groups)

        gen_optim = instantiate(
            self.hparams.optim.generator.optimizer, params=gen_params
        )

        # Same handling for generator scheduler
        scheduler_config = self.hparams.optim.generator.scheduler

        if hasattr(scheduler_config, "schedulers"):
            # Create the nested schedulers first
            nested_schedulers = []
            for sched_cfg in scheduler_config.schedulers:
                nested_scheduler = instantiate(sched_cfg, optimizer=gen_optim)
                nested_schedulers.append(nested_scheduler)

            # Create the SequentialLR with the nested schedulers
            gen_sched = instantiate(
                scheduler_config, optimizer=gen_optim, schedulers=nested_schedulers
            )
        else:
            # Regular scheduler instantiation
            gen_sched = instantiate(scheduler_config, optimizer=gen_optim)

        return [disc_optim, gen_optim], [
            {"scheduler": disc_sched, "interval": "step"},
            {"scheduler": gen_sched, "interval": "step"},
        ]
