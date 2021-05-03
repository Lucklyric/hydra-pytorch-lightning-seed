#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File  : mnist_data_module.py
# Author: Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
# Date  : 02.05.2021
import logging

import numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import random_split
from torchvision.datasets.mnist import MNIST

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    """PyTorch Dataloader worker init function for setting different seed to each worker process
    Args:
            worker_id: the id of current worker process
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class DummyMNISTWrapper(torch.utils.data.Dataset):

    def __init__(self, wrapped_dataset, noise_level=0.1, *args, **kwargs):
        self.base_dataset = wrapped_dataset
        self.noise_level = noise_level

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        clean = np.expand_dims(np.array(self.base_dataset[index][0]),
                               0)  #[1,28,28]
        noisy = np.random.normal(0, scale=self.noise_level,
                                 size=clean.shape) + clean

        return {'clean': clean.astype(np.float32), 'noisy': noisy.astype(np.float32)}


class DummyMNISTDataModule(LightningDataModule):

    def __init__(self,
                 train_batch_size=1,
                 val_batch_size=1,
                 test_batch_size=1,
                 train_num_workers=4,
                 val_num_workers=4,
                 test_num_workers=4,
                 noise_level=0.1,
                 data_dir='',
                 train=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers

        if train:
            dataset = MNIST(data_dir, train=True, download=True)
            mnist_train, mnist_val = random_split(dataset, [55000, 5000])
            self.train_dataset = DummyMNISTWrapper(mnist_train,
                                                   noise_level=noise_level)
            self.val_dataset = DummyMNISTWrapper(mnist_val,
                                                 noise_level=noise_level)
            logger.info(
                f'len of train examples {len(self.train_dataset)}, len of val examples {len(self.val_dataset)}'
            )
        else:
            mnist_test = MNIST(data_dir, train=False, download=True)
            self.test_dataset = DummyMNISTWrapper(mnist_test,
                                                  noise_level=noise_level)
            logger.info(f'len of test examples {len(self.test_dataset)}')

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.test_num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn)
        return test_loader
