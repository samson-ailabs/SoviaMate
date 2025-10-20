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

"""A flexible training pipeline for various tasks."""

import warnings

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch

L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="configs/training")
def main(config: DictConfig):
    r"""Main function to run the training pipeline."""

    task = instantiate(config.task, _recursive_=False)

    callbacks = None
    if config.get("callbacks") is not None:
        callbacks = [instantiate(cfg) for cfg in config["callbacks"].values()]

    loggers = None
    if config.get("loggers") is not None:
        loggers = [instantiate(cfg) for cfg in config["loggers"].values()]

    trainer = L.Trainer(callbacks=callbacks, logger=loggers, **config.trainer)
    trainer.fit(task)


if __name__ == "__main__":
    main()
