
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from glob import glob
from helpers.helpers import *
import numpy as np
import pandas as pd
import copy
import pickle

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def matpadder(x, max_in=512):
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    shape = x.shape
    # delta1 = max_in - shape[0]
    delta2 = max_in - shape[-1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out



def preprocess_asinh(x, mu, sigma, lam=0.1):
    z = (x - mu) / sigma
    return torch.asinh(z / lam)              # y  (bounded ~ [-asinh, asinh])

def inv_preprocess_asinh(y, mu, sigma, lam=0.1):
    z = lam * torch.sinh(y)
    return z * sigma + mu



class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, zoo_root, zoo_split, length, resolution, to_image, in_channel, topk,
                 transform=None, scale=1.0, lamda=0.1):
        super(ZooDataset, self).__init__()
        self.zoo_root = zoo_root
        self.zoo_split = zoo_split
        self.length = length
        self.resolution = resolution
        self.to_image = to_image
        self.in_channel = in_channel
        self.transform = transform
        self.lamda = lamda
        self.topk = topk
        self.scale = scale
        #min:-12.5625 max:11.6875--mean:-0.00015318099758587778 std:0.14582251012325287
        #shape:torch.Size([124421, 196608]) min:-10.1875 max:11.1875--mean:3.373968866071664e-05 std:0.07287021726369858

        self.data_dir = os.path.join(self.zoo_root, self.zoo_split)
        data = glob(f"{self.zoo_root}/{self.zoo_split}/*/*.pt")
        print(f"{len(data)} files found in {self.zoo_root}/{self.zoo_split}")
        if self.topk is not None:
            data = data[:self.topk]
        self.files_list = data
        # self.x_min = -12.5625
        # self.x_max = 11.6875
        # self.x_min=-10.1875
        # self.x_max =11.1875
        # self.mu = -0.00015318099758587778
        # self.std = 0.14582251012325287
        self.x_min=-10.1875
        self.x_max=11.1875
        self.mu=0
        self.std=1.0


    def __len__(self):
            return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fl = self.files_list[idx]
        weight = torch.load(fl , map_location='cpu',weights_only=False)
        # if self.transform is not None:
        #     weight = self.transform(weight)
        weight = torch.tensor(weight.cpu().contiguous().float().numpy())
        weight = (weight-weight.mean())/(weight.std())
        # weight = preprocess_asinh(weight, self.mu, self.std, self.lamda)
        # weight = 2*(weight-self.x_min)/(self.x_max-self.x_min)-1
        # weight = (weight - self.mu) / self.std
        if self.to_image:
            weight = weight.contiguous().reshape(self.in_channel,  self.resolution, self.resolution)
        else:
            weight = weight.contiguous().reshape(self.length,)
        weight = weight/self.scale
        sample = {'weight': weight, 'chunk_idx':[], 'layer_info':[]}
        return sample



