import os

import torch
import torch.nn.functional as F
import yaml
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


class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, root='zoodata', dataset="joint", split='train', topk=None, scale=1.0, transform=None, normalize=False,
                 max_len=10240):
        super(ZooDataset, self).__init__()
        #1960513
        self.topk = topk
        self.max_len = max_len
        self.split = split
        self.dataset = dataset
        self.normalize = normalize
        self.chunk_size = max_len
        # self.max_len = 2560*64
        self.scale=scale
#51380224==4096 x 12544

        # datapath = os.path.join(root, f'llmdata/gemma-3-4b-it_full_mat_mlp.pt')  # 262144
        # datapath = os.path.join(root, f'llmdata/gemma-3-4b-it_full_mat_self_attn.pt')  # 262144
        # datapath = os.path.join(root, f'llmdata/gemma-3-4b-it_self_attn_mat.pt')  # 262144
        datapath = os.path.join(root, f'llmdata/gemma-3-4b-it_full_mat.pt')  # 262144
        # datapath = os.path.join(root, f'llmdata/gemma-3-4b-it_full_mat_gate_proj.pt')  # 262144
        # datapath = os.path.join(root, f'llmdata/gemma-3-4b-it_full_mat_mlp.pt')  # 262144
        #

        # self.transform = transform
        # self.transform = transforms.Lambda(RandomSwapTransform(p=0.25))
        # self.transform = # Simple lambda transform that adds Gaussian noise
        sigma = 0.001
        max_noise = 3 * sigma
        p_noise = 0.5  # 50% chance to apply noise

        self.transform = transforms.Lambda(
            lambda x: x + torch.clamp(sigma * torch.randn_like(x), -max_noise, max_noise)
            if random.random() < p_noise else x
        )
        # sigma = 0.001
        # max_noise = 3 * sigma

        # self.transform = transforms.Lambda(
        #     lambda x: x + torch.clamp(sigma * torch.randn_like(x), -max_noise, max_noise)
        # )
        # self.transform = transforms.Lambda(lambda x: x + sigma * torch.randn_like(x))
        data, self.targets = self.load_data(datapath, dataset=dataset)


        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')
        # data, md, ir = robust_scale(data)
        # print(data[0][:20])
        # data = 2 * (data - x_min) / (x_max - x_min) - 1
        mu = data.mean()
        std = data.std()

        print(f'============{std}==============={mu}=============')
        # exit()
        # data = (data.float()-mu)/std

        # exit()
        self.data = data.detach().cpu()
        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            weight = self.transform(weight)

        weight= weight/self.scale
        # sample = {'weight': weight, 'dataset': label}
        return weight, label
    def load_data(self, file, dataset='joint'):
        data = torch.load(file)
        wl = []
        layer_ids=[]
        chunk_ids =[]
        keys = list(data)
        print(f'--number --item--{len(keys)}')
        # exit()

        keys =keys[:-1]

        for jj, k in enumerate(keys):
            ii = 0
            # w = data[k]
            w = data[k].reshape(1, -1)
            print(w.shape)
            if self.normalize == "z_score":
                u = torch.mean(w)
                v = torch.std(w)
                w = (w - u) / v
            elif self.normalize == "min_max":
                x_max = torch.max(w)
                x_min = torch.min(w)
                xdiff = x_max - x_min
                w = 2 * (w - x_min) / xdiff - 1
            w = pad_to_chunk_multiple(w, chunk_size=self.chunk_size)
            w = w.reshape(1, -1)
            w = torch.split(w, split_size_or_sections=self.max_len, dim=-1)
            y = torch.tensor(list(range(len(w))))
            chunk_ids.extend(y.reshape(-1).tolist())
            lb = [jj] * len(y)
            print(len(y))
            layer_ids.extend(lb)
            w = torch.cat(w, dim=0)

            if self.topk is not None:
                if self.topk > 0:
                    w = w[:self.topk]
                    wl.append(w)
            else:
                wl.append(w)
        data = torch.cat(wl, dim=0)
        labels = torch.tensor(layer_ids, dtype=torch.long).reshape(-1, 1)
        clabels = torch.tensor(chunk_ids, dtype=torch.long).reshape(-1, 1)
        # print(max(labels), max(clabels))
        targets = torch.cat([labels, clabels], dim=-1)

        return data, targets


