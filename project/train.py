#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
# Date  : 02.05.2021
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

log = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name='train_config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)

    # ------------
    # data
    # ------------
    data_module = hydra.utils.instantiate(cfg.data)


    # ------------
    # model
    # ------------
    model = hydra.utils.instantiate(cfg.model)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(**(cfg.pl_trainer), checkpoint_callback=True)
    log.info('run training...')
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=[val_dataloader])


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(e)
        exit(1)
