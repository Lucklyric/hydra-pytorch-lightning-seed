#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File  : dncnn.py
# Author: Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
# Date  : 02.05.2021
import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from mrc_insar_common.util.pt.init import weight_init

logger = logging.getLogger(__name__)


class DnCNN(pl.LightningModule):

    def __init__(self,
                 lr=0.0003,
                 channels=3,
                 num_layers=7,
                 num_features=64,
                 *args,
                 **kwargs):
        super().__init__()
        self.lr = lr
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(
            nn.Conv2d(in_channels=channels,
                      out_channels=num_features,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=num_features,
                          out_channels=num_features,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=num_features,
                      out_channels=channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)

        self.dncnn.apply(weight_init)

    def forward(self, x):
        x = self.dncnn(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        clean = batch['clean']
        noisy = batch['noisy']

        # run forward
        out = noisy - self.forward(noisy)

        loss = ((out - clean)**2).mean()

        self.log('my_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return loss

    def validation_step(self, x, batch_idx):
        example_root = f'{self.trainer.log_dir}/{self.trainer.current_epoch}:{self.global_step}:{batch_idx}'
        clean = x['clean']
        noisy = x['noisy']
        with torch.no_grad():
            out = (noisy - self.forward(noisy))

            clean_sample = clean[0].detach().cpu().numpy().squeeze()
            noisy_sample = noisy[0].detach().cpu().numpy().squeeze()
            filt_sample = out[0].detach().cpu().numpy().squeeze()

            metric_dict = {'val_loss': ((out - clean)**2).mean()}

            self.log_dict(metric_dict)
            # plot first sample
            clean_example_path = f'{example_root}:{batch_idx}.0.clean.png'
            noisy_example_path = f'{example_root}:{batch_idx}.0.noisy.png'
            filt_example_path = f'{example_root}:{batch_idx}.0.filt.png'
            plt.imsave(clean_example_path,
                       clean_sample,
                       cmap='gray',
                       vmin=0,
                       vmax=1)
            plt.imsave(noisy_example_path,
                       noisy_sample,
                       cmap='gray',
                       vmin=0,
                       vmax=1)
            plt.imsave(filt_example_path,
                       filt_sample,
                       cmap='gray',
                       vmin=0,
                       vmax=1)

    def test_step(self, x, batch_idx):
        clean = x['clean']
        noisy = x['noisy']
        with torch.no_grad():
            out = noisy - self.forward(noisy)
            out_np = out.detach().cpu().numpy();
            clean_np = clean.detach().cpu().numpy();
            mse = np.square(out_np - clean_np).mean()
            metric_dict = {
                'mse': mse 
            }
            self.log_dict(metric_dict)
        return mse

    def test_epoch_end(self, outputs):
        # for test_epoch_end showcase, we record batch mse and batch std
        all_res = np.asarray(outputs)
        self.log_dict({'b_mse': np.mean(all_res), 'b_std': np.std(all_res)})
