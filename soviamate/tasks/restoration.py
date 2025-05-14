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

"""“Audio Neural Codec Models for learning audio discrete tokens”"""

import itertools
from typing import Tuple

import lightning as L
from hydra.utils import instantiate

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SEGMENT_SIZE = 40960


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
        self.audio_decoder = instantiate(self.hparams.model.audio_decoder)
        self.audio_quantizer = instantiate(self.hparams.model.audio_quantizer)

        self.discriminator = instantiate(self.hparams.model.discriminator)

        self.audio_loss = instantiate(self.hparams.model.audio_loss)
        self.adv_loss = instantiate(self.hparams.model.adv_loss)

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
        target_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        encoder_outputs, encoder_lengths = self.audio_encoder(
            input_audios, input_lengths
        )

        encoder_outputs = self.audio_quantizer(encoder_outputs)

        decoder_outputs, decoder_lengths = self.audio_decoder(
            encoder_outputs, encoder_lengths
        )

        if target_lengths is not None:
            decoder_outputs = F.pad(
                decoder_outputs,
                (0, 0, 0, target_lengths.max() - decoder_outputs.size(1)),
                value=0,
            )
            decoder_lengths = target_lengths

        return decoder_outputs, decoder_lengths

    def training_step(self, batch: Tuple[torch.Tensor, ...]):

        input_audios, input_lengths, _, _, target_audios, target_lengths, _, _ = batch
        output_audios, _ = self.forward(input_audios, input_lengths, target_lengths)

        output_audios = output_audios.transpose(1, 2)
        target_audios = target_audios.transpose(1, 2)

        disc_optim, gen_optim = self.optimizers()
        disc_sched, gen_sched = self.lr_schedulers()

        # Train discriminator
        self.toggle_optimizer(disc_optim)

        outputs, targets = self._get_random_segments(
            output_audios, target_audios, target_lengths
        )

        outputs = self.discriminator(outputs.detach())
        targets = self.discriminator(targets.detach())

        disc_loss = self.adv_loss(outputs, targets)
        self.manual_backward(disc_loss)

        disc_optim.step()
        disc_optim.zero_grad()

        if self.trainer.is_last_batch:
            disc_sched.step()

        self.untoggle_optimizer(disc_optim)

        # Train Generator
        self.toggle_optimizer(gen_optim)

        audio_loss = self.audio_loss(output_audios, target_audios, target_lengths)

        outputs, _ = self._get_random_segments(
            output_audios, target_audios, target_lengths
        )

        outputs = self.discriminator(outputs)
        gen_loss = self.adv_loss(outputs)

        loss = 2.0 * audio_loss + gen_loss
        self.manual_backward(loss)

        gen_optim.step()
        gen_optim.zero_grad()

        if self.trainer.is_last_batch:
            gen_sched.step()

        self.untoggle_optimizer(gen_optim)

        self.log_dict(
            {
                "train_audio_loss": audio_loss,
                "train_gen_loss": gen_loss,
                "train_disc_loss": disc_loss,
            },
            sync_dist=True,
        )

        self.log("train_loss", loss, sync_dist=True, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, ...]):
        input_audios, input_lengths, _, _, target_audios, target_lengths, _, _ = batch
        output_audios, _ = self.forward(input_audios, input_lengths, target_lengths)

        output_audios = output_audios.transpose(1, 2)
        target_audios = target_audios.transpose(1, 2)

        audio_loss = self.audio_loss(output_audios, target_audios, target_lengths)

        outputs, _ = self._get_random_segments(
            output_audios, target_audios, target_lengths
        )

        outputs = self.discriminator(outputs)
        gen_loss = self.adv_loss(outputs)

        loss = 2.0 * audio_loss + gen_loss

        self.log_dict(
            {
                "val_audio_loss": audio_loss,
                "val_gen_loss": gen_loss,
            },
            sync_dist=True,
        )

        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

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

        disc_optim = instantiate(self.hparams.optim.optimizer, params=disc_params)
        disc_sched = instantiate(self.hparams.optim.scheduler, optimizer=disc_optim)

        gen_params = itertools.chain(
            self.audio_encoder.parameters(), self.audio_decoder.parameters()
        )

        gen_optim = instantiate(self.hparams.optim.optimizer, params=gen_params)
        gen_sched = instantiate(self.hparams.optim.scheduler, optimizer=gen_optim)

        return [disc_optim, gen_optim], [disc_sched, gen_sched]
