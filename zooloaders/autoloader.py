import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler

from zoodatasets.ddp_datasets import ZooDataset


def my_collate(batch):
    sample = {}
    data = [item['weight'] for item in batch]
    sample['weight'] = torch.cat(data, 0)
    return sample

class ZooDataModule(pl.LightningDataModule):
    def __init__(self, zoo_root, zoo_name, batch_size=32, num_workers=4, scale=1.0, topk=None, normalize=False,
                 tgt=None, exd=None, length=65536, transform=None, split='train',toimage_shape=False, sizes=[3, 256],
                 zoo_file='zoo_config.simple_config.yaml',  blk_mul=1, take_k=1):
        super().__init__()
        self.zoo_root = zoo_root
        self.zoo_name = zoo_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.topk = topk
        self.normalize = normalize
        self.scale=scale
        self.length=length
        self.tgt = tgt
        self.exd = exd
        self.transform = transform
        self.split = split
        self.toimage_shape = toimage_shape
        self.size = sizes
        self.zoo_file = zoo_file
        self.blk_mul = blk_mul
        self.take_k = take_k


    def prepare_data(self):
        pass
        # datasets.CIFAR10(self.data_root, train=True, download=True)
        # datasets.CIFAR10(self.data_root, train=False, download=True)

    def setup(self, stage):

        if stage == "fit":
            self.trainset = ZooDataset(zoo_root=self.zoo_root, zoo_name=self.zoo_name, split='train',
                                       scale=self.scale, topk=self.topk, normalize=self.normalize,
                                       tgt=self.tgt, exd=self.exd, length=self.length, toimage_shape=self.toimage_shape,
                                       sizes=self.size, zoo_file=self.zoo_file, transform=self.transform,
                                       take_k=self.take_k)
            self.valset = ZooDataset(zoo_root=self.zoo_root, zoo_name=self.zoo_name, split='train',
                                       scale=self.scale, topk=5, normalize=self.normalize,
                                       tgt=self.tgt, exd=self.exd, length=self.length, toimage_shape=self.toimage_shape,
                                       sizes=self.size, zoo_file=self.zoo_file, transform=self.transform,
                                     take_k=self.take_k)

        if stage == "test":
            pass
            # self.testset = ZooDataset(zoo_root=self.zoo_root, zoo_name=self.zoo_name, split='train',
            #                            scale=self.scale, topk=10, normalize=self.normalize,
            #                            tgt=self.tgt, exd=self.exd)
        if stage == "predict":
            pass
            # pass

    def train_dataloader(self):
        # sampler = DistributedSampler(self.trainset, shuffle=True)
        # sampler = DistributedSampler(self.trainset)
        return DataLoader(
            self.trainset,
            # sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            # collate_fn=my_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            # collate_fn=my_collate,
        )

    def test_dataloader(self):
        pass
        # return DataLoader(
        #     self.testset,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False,
        # )

    def predict_dataloader(self):
        pass
        # return DataLoader(
        #     self.cifar10_predict,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False,
        # )

