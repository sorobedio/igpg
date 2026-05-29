import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric

# from data_utils.zoodataset import ZooDataset
# from data_utils.hyperzoodata import ZooDataset
# from zoodatasets.ldmdatasets import ZooDataset
# from zoodatasets.layerdatasets import ZooDataset
# from zoodatasets.basedatasets import ZooDataset
from zoodatasets.cond_datasets import ZooDataset
# from zoodatasets.vit_dataset_chunk import ZooDataset


class ZooDataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_dir, data_root, batch_size, num_workers, topk, normalize, scale=1.0,
                 tgt=None, exd='norm', length=65536, channel =1, input_size=32,):
        super().__init__()
        self.data_dir = data_dir
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.topk = topk
        self.normalize = normalize
        self.tgt = tgt
        self.exd = exd
        self.length = length
        self.channel = channel
        self.input_size = input_size
        # print(dataset)

        # self.num_sample = num_sample
        self.scale=scale

        # self.transform = []

    def prepare_data(self):
        pass
    #     datasets.CIFAR10(self.data_root, train=True, download=True)
    #     datasets.CIFAR10(self.data_root, train=False, download=True)

    def setup(self, stage):

        if stage == "fit":
            # ZooDataset(root='modelzoos', dataset="DeepSeek-R1-Distill-Qwen-14B", split='train',
            #               scale=1.0, normalize=None, tgt=None, exd='norm', length=65536)
            self.trainset = ZooDataset(root=self.data_dir, dataset=self.dataset, split='train', topk=self.topk,
                                       normalize=self.normalize, scale=self.scale, tgt=self.tgt, exd=self.exd,
                                       length=self.length, channel=self.channel, input_size=self.input_size)
            self.valset = ZooDataset(root=self.data_dir, dataset=self.dataset, split='train', topk=10,
                                       normalize=self.normalize, scale=self.scale, tgt=self.tgt, exd=self.exd,
                                       length=self.length, channel=self.channel, input_size=self.input_size)

        if stage == "test":
            self.testset = ZooDataset(root=self.data_dir, dataset=self.dataset, split='train', topk=s10,
                                       normalize=self.normalize, scale=self.scale, tgt=self.tgt, exd=self.exd,
                                       length=self.length, channel=self.channel, input_size=self.input_size)

        if stage == "predict":
            self.cifar10_predict = datasets.CIFAR10(self.data_root, train=False, transform=self.transform)
            # pass

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    # dataloader to evaluate the reconstruction performance on model zoo.
    def predict_dataloader(self):
        return DataLoader(
            self.cifar10_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

