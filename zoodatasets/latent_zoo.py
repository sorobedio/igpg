import os

import torch
import torch.nn.functional as F
import yaml
from sympy.stats import sample
from torch.utils.data import Dataset
from glob import glob
import math
import torchvision.transforms as transforms
import torch
import random


def robust_scale(weights: torch.Tensor):
    # Compute the median and the 25th and 75th percentiles using torch.quantile
    median = torch.median(weights)
    q1 = torch.quantile(weights, 0.25)
    q3 = torch.quantile(weights, 0.75)
    iqr = q3 - q1

    # Avoid division by zero if IQR is zero
    if iqr == 0:
        raise ValueError("IQR is zero, cannot robustly scale the weights.")

    # Compute the robust scaled weights
    scaled_weights = (weights - median) / iqr
    return scaled_weights, median, iqr


def inverse_robust_scale(scaled_weights: torch.Tensor, median: torch.Tensor, iqr: torch.Tensor):
    # Inverse the robust scaling transformation
    return scaled_weights * iqr + median




def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


class RandomSwapTransform:
    def __init__(self, p=0.5):
        """
        Args:
            p (float): Probability of swapping the first and second halves of the input vector.
        """
        self.p = p

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Input vector of shape (N,)
        Returns:
            torch.Tensor: Transformed vector with swapped halves (if applicable)
        """
        if torch.rand(1).item() < self.p:
            mid = x.shape[0] // 2
            return torch.cat([x[mid:], x[:mid]])
        return x

def matpadder(x, max_in=512):
    shape = x.shape
    # delta1 = max_in - shape[0]
    if len(shape)<2:
        x =x.unsqueeze(0)
        shape = x.shape
    delta2 = max_in - shape[1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out
def pad_to_chunk_multiple(x, chunk_size):
    shape = x.shape
    if len(shape)<2:
        x =x.unsqueeze(0)
        shape = x.shape
    max_in = chunk_size*math.ceil(shape[1]/chunk_size)
    if max_in> shape[1]:
        delta1 = max_in - shape[1]
        x =F.pad(x, (0, delta1, 0, 0), "constant", 0)
    return x
class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, root='zoodata', split='train', topk=None, scale=1.0, transform=None, normalize=False,
                 max_len=10240):
        super(ZooDataset, self).__init__()
        #1960513
        self.topk = topk
        # self.max_len = max_len
        self.split = split
        # self.dataset = dataset
        self.normalize = normalize
        self.chunk_size = max_len
        self.max_len = max_len
        self.scale=scale

        #
        file1 = os.path.join(root, f'gemma_4b_latent_kv.pt')  # 262144
        file2 = os.path.join(root, f'gemma_1b_latent_kv.pt')  # 262144 gemma-3-4b-it_self_attn.pt


        # self.transform = transform
        # self.transform = transforms.Lambda(lambda x: torch.asinh(x))
        self.transform = transform
        # sigma = 0.0001
        # max_noise = 3 * sigma
        # p_noise = 0.5  # 50% chance to apply noise
        # self.transform = transforms.Lambda(
        #     lambda x: x + torch.clamp(sigma * torch.randn_like(x), -max_noise, max_noise)
        # )
        # self.transform = transform
        # sigma = 0.001
        # max_noise = 3 * sigma
        # p_noise = 0.5  # 50% chance to apply noise
        #
        # self.transform = transforms.Lambda(
        #     lambda x: x + torch.clamp(sigma * torch.randn_like(x), -max_noise, max_noise)
        #     if random.random() < p_noise else x
        # )
        self.transform = RandomSwapTransform(p=0.5)
        data, self.target= self.load_data(file1, file2=file2)

        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')
        # data, md, ir = robust_scale(data)
        # print(data[0][:20])
        # data = 2 * (data - x_min) / (x_max - x_min) - 1
        mu = data.mean()
        std = data.std()

        print(f'============{std}==============={mu}=============')
        # exit()
        # data = (data.float()-mu)/std
        # data = data.repeat(2, 1)
        # exit()
        self.data = data.detach().cpu()


        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')
        print(f'===============dataset size=={self.target.shape}======max={self.target.max()}======={self.target.min()}==========')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx]
        taget = self.target[idx]
        if self.transform:
            weight = self.transform(weight)
            taget = self.transform(taget)

        weight= weight/self.scale
        taget=taget/self.scale
        # sample = {'weight': weight, 'dataset': []}
        return weight, taget
    def load_data(self, file, file2):
        data = torch.load(file)
        tagets =torch.load(file2)
        keys = list(tagets.keys())
        wl = []
        tl =[]
        for k in keys:
            wt = tagets[k]
            ws = data[k]
            ws = pad_to_chunk_multiple(ws, chunk_size=self.chunk_size)
            ws = torch.split(ws, split_size_or_sections=self.chunk_size, dim=-1)
            ws = torch.cat(ws, dim=0)
            wl.append(ws)

            wt = pad_to_chunk_multiple(wt, chunk_size=self.chunk_size)
            wt = torch.split(wt, split_size_or_sections=self.chunk_size, dim=-1)
            wt = torch.cat(wt, dim=0)
            tl.append(wt)
        data = torch.cat(wl, dim=0)
        tagets = torch.cat(tl, dim=0)
        return data, tagets


