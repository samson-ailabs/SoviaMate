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
import random
from typing import Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from soviamate.modules.discriminator import SEGMENT_SIZE
from soviamate.utils.helper import make_padding_mask


class _TimbreSuppression(nn.Module):
    """Adversarial timbre-suppression head with gradient reversal.

    Stats-pools encoder features (mean + std, matching speaker verifiers'
    own pooling) and regresses a speaker embedding through a 2-layer MLP
    with dropout. Gradient reversal flips the sign on the encoder side so
    the encoder is pushed to produce timbre-uninformative features.

    Args:
        input_dim (int): Encoder feature dimension.
        output_dim (int): Speaker embedding dimension.
        hidden_dim (int): MLP hidden dimension. Default: ``512``.
        dropout (float): Dropout probability inside the MLP. Default: ``0.1``.
        coef (float): Gradient reversal coefficient. Default: ``1.0``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        coef: float = 1.0,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.coef = float(coef)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        r"""Stats-pool, reverse gradient, project.

        Args:
            features (Tensor): Encoder features, shape ``(B, T, D)``.
            lengths (Tensor): Per-sample valid frame counts, shape ``(B,)``.

        Returns:
            Tensor: Predicted speaker embeddings, shape ``(B, output_dim)``.
        """
        valid_mask = (~make_padding_mask(lengths)).to(features.dtype).unsqueeze(-1)
        frame_counts = valid_mask.sum(dim=1).clamp(min=1)

        mean = (features * valid_mask).sum(dim=1) / frame_counts
        centered = (features - mean.unsqueeze(1)) * valid_mask
        variance = centered.square().sum(dim=1) / frame_counts
        std = (variance + 1e-5).sqrt()

        stats = torch.cat([mean, std], dim=-1)
        return self.proj((1.0 + self.coef) * stats.detach() - self.coef * stats)


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

        # Adversarial timbre-suppression head, shared between pre-quant and post-quant probes.
        if hasattr(model, "timbre_grl"):
            self.timbre_grl = _TimbreSuppression(**model.timbre_grl)
        else:
            self.timbre_grl = None

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
        speaker_embeddings: torch.Tensor,
        target_audio_lengths: torch.Tensor,
        apply_splice_out: bool = False,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # Encode source (augmented) audios
        source_features, source_feature_lengths = self.audio_encoder(
            source_audios, source_audio_lengths
        )

        # Quantize encoded features into latent representations
        quantized_features, quantized_feature_lengths = self.audio_quantizer(
            source_features, source_feature_lengths
        )

        # Auxiliary ASR: 50% post-quant (trains quantizer to preserve content),
        # 50% pre-quant (keeps text decoder compatible with streaming inference)
        output_tokens, output_token_lengths = None, None
        if self.text_decoder is not None:
            if self.training and random.random() < 0.5:
                asr_features = quantized_features
                asr_feature_lengths = quantized_feature_lengths
            else:
                asr_features = source_features
                asr_feature_lengths = source_feature_lengths

            if apply_splice_out and self.splice_out is not None:
                asr_features, asr_feature_lengths = self.splice_out(
                    asr_features, asr_feature_lengths
                )

            output_tokens, output_token_lengths = self.text_decoder(
                asr_features, asr_feature_lengths
            )

        # Decode to audio with speaker conditioning
        max_output_length = target_audio_lengths.max().item()
        output_audios, output_audio_lengths = self.audio_decoder(
            quantized_features,
            quantized_feature_lengths,
            speaker_embeddings,
            max_output_length,
        )

        return (
            source_features,
            source_feature_lengths,
            quantized_features,
            quantized_feature_lengths,
            output_audios,
            output_audio_lengths,
            output_tokens,
            output_token_lengths,
        )

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        (
            source_audios,
            source_audio_lengths,
            speaker_embeddings,
            target_audios,
            target_audio_lengths,
            target_tokens,
            target_token_lengths,
        ) = batch

        (
            source_features,
            source_feature_lengths,
            quantized_features,
            quantized_feature_lengths,
            output_audios,
            _,
            output_tokens,
            output_token_lengths,
        ) = self.forward(
            source_audios,
            source_audio_lengths,
            speaker_embeddings,
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

        timbre_loss = 0.0
        if self.timbre_grl is not None:
            timbre_loss = self._compute_timbre_loss(
                source_features,
                source_feature_lengths,
                quantized_features,
                quantized_feature_lengths,
                speaker_embeddings,
            )

        train_loss = 2.0 * audio_loss + gen_loss + 3.0 * text_loss + timbre_loss
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

        if timbre_loss > 0 and self.timbre_grl is not None:
            log_dict["train_timbre_loss"] = timbre_loss

        if text_loss > 0 and self.text_loss is not None:
            log_dict["train_text_loss"] = text_loss

        self.log_dict(log_dict, sync_dist=True)
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, ...]):
        (
            source_audios,
            source_audio_lengths,
            speaker_embeddings,
            target_audios,
            target_audio_lengths,
            target_tokens,
            target_token_lengths,
        ) = batch

        (_, _, _, _, output_audios, _, output_tokens, output_token_lengths) = (
            self.forward(
                source_audios,
                source_audio_lengths,
                speaker_embeddings,
                target_audio_lengths,
                apply_splice_out=False,
            )
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

        val_loss = 2.0 * audio_loss + gen_loss + 3.0 * text_loss
        self.log("val_loss", val_loss, sync_dist=True, prog_bar=True)

        log_dict = {"val_audio_loss": audio_loss, "val_gen_loss": gen_loss}

        if text_loss > 0 and self.text_loss is not None:
            log_dict["val_text_loss"] = text_loss

        self.log_dict(log_dict, sync_dist=True)

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

    def _compute_timbre_loss(
        self,
        source_features: torch.Tensor,
        source_feature_lengths: torch.Tensor,
        quantized_features: torch.Tensor,
        quantized_feature_lengths: torch.Tensor,
        speaker_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Adversarial timbre-suppression loss averaged over pre- and post-quant features.

        Args:
            source_features: Encoder features before quantization, shape (B, T, D).
            source_feature_lengths: Lengths of encoder features, shape (B,).
            quantized_features: Quantized features after vector quantization, shape (B, T, D).
            quantized_feature_lengths: Lengths of quantized features, shape (B,).
            speaker_embeddings: Ground-truth speaker embeddings, shape (B, E).

        Returns:
            Tensor: Combined timbre loss for encoder and quantizer, a scalar.
        """
        target = speaker_embeddings.detach()

        pred_encoder = self.timbre_grl(source_features, source_feature_lengths)
        pred_quantizer = self.timbre_grl(quantized_features, quantized_feature_lengths)

        sim_encoder = F.cosine_similarity(pred_encoder, target, dim=-1)
        sim_quantizer = F.cosine_similarity(pred_quantizer, target, dim=-1)

        loss_encoder = 1.0 - sim_encoder.square().mean()
        loss_quantizer = 1.0 - sim_quantizer.square().mean()

        return 0.5 * (loss_encoder + loss_quantizer)

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

        # Add timbre head parameters if available
        if self.timbre_grl is not None:
            gen_param_groups.append(self.timbre_grl.parameters())

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

        This method saves only the production-relevant components along with hyperparameters.
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

        # Optional components for text decoding
        if self.text_decoder is not None:
            model_weights["text_decoder"] = self.text_decoder.state_dict()

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
