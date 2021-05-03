#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File  : test.py
# Author: Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
# Date  : 02.05.2021

import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import json

log = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name='test_config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)

    # ------------
    # data
    # ------------
    data_module = hydra.utils.instantiate(cfg.data) 
    test_dataloader = data_module.test_dataloader()

    # ------------
    # model
    # ------------
    model = hydra.utils.instantiate(cfg.model) 
    # model.load_from_checkpoint(cfg.checkpoint)
    model.load_state_dict(torch.load(cfg.checkpoint_path)['state_dict'])

    trainer = pl.Trainer(**(cfg.pl_trainer))

    # ------------
    # testing
    # ------------
    result = trainer.test(model, test_dataloaders=test_dataloader)
    log.info(result)
    with open(f'{trainer.log_dir}/out', 'w') as f:
        f.write(json.dumps(str(result),indent=4))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(e)
        exit(1)
