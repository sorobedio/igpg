import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler
from zoodatasets.autoloader import ZooDataset


def my_collate(batch):
    # sample = {}
    data = [item['weight'] for item in batch]
    # text = [item['layer_info'] for item in batch['dataset']]
    text = [item for sub in batch['dataset']['layer_info'] for item in sub]

    batch['weight'] = torch.cat(data, 0)
    batch['dataset']['layer_info'] = text
    return batch

class ZooDataModule(pl.LightningDataModule):
    def __init__(self, zoo_root, zoo_split, batch_size=32, num_workers=4, scale=1.0, topk=None,
                 resolution=64, length=65536, transform=None,in_channel=1, to_image=False,normalize=None,
                 ):
        super().__init__()
        self.zoo_root = zoo_root
        self.zoo_split = zoo_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.in_channel = in_channel
        self.transform = transform
        self.to_image = to_image
        self.topk = topk
        self.scale=scale
        self.length=length
        self.normalize = normalize

    def prepare_data(self):
        pass
        # datasets.CIFAR10(self.data_root, train=True, download=True)
        # datasets.CIFAR10(self.data_root, train=False, download=True)

    def setup(self, stage):
        # ZooDataset(zoo_root=zoo_root, zoo_split=zoo_split, length=length, resolution=resolution,
        #            to_image=to_image, in_channel=in_channel, topk=topk)

        if stage == "fit":
            self.trainset = ZooDataset(zoo_root=self.zoo_root, zoo_split=self.zoo_split, scale=self.scale, topk=self.topk,
                                       length=self.length, to_image=self.to_image,in_channel=self.in_channel,
                                       resolution=self.resolution, normalize=self.normalize,
                                       transform=self.transform)

            self.valset = ZooDataset(zoo_root=self.zoo_root, zoo_split=self.zoo_split, scale=self.scale, topk=self.topk,
                                       length=self.length, to_image=self.to_image,in_channel=self.in_channel,
                                     resolution=self.resolution, normalize=self.normalize,
                                       transform=self.transform)
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

