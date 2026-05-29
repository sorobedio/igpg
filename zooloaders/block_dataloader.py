import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler

from zoodatasets.block_datasets import ZooDataset

def my_collate(batch):
    sample = {}
    data = [item['weight'] for item in batch]
    sample['weight'] = torch.cat(data, 0)
    return sample

class ZooDataModule(pl.LightningDataModule):
    def __init__(self, zoo_root, zoo_split, length, resolution, to_image, in_channel, topk,scale=1.0,
                 batch_size=4, num_workers=16):
        super().__init__()
        self.zoo_root = zoo_root
        self.zoo_split = zoo_split
        self.length = length
        self.resolution = resolution
        self.to_image = to_image
        self.in_channel = in_channel
        self.topk = topk
        self.scale = scale
        self.batch_size = batch_size
        self.num_workers = num_workers



    def prepare_data(self):
        pass
        # datasets.CIFAR10(self.data_root, train=True, download=True)
        # datasets.CIFAR10(self.data_root, train=False, download=True)

    def setup(self, stage):

        if stage == "fit":
            self.trainset = ZooDataset(zoo_root=self.zoo_root, zoo_split=self.zoo_split, length=self.length,
                                       resolution=self.resolution, to_image=self.to_image,
                                       in_channel=self.in_channel, topk=self.topk,scale=self.scale)
            self.valset = ZooDataset(zoo_root=self.zoo_root, zoo_split=self.zoo_split, length=self.length,
                                       resolution=self.resolution, to_image=self.to_image,
                                       in_channel=self.in_channel, topk=1000,scale=self.scale)
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
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

