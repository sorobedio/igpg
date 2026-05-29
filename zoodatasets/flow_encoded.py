import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from glob import glob
import math
import torchvision.transforms as transforms
import torch


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

        # datapath = os.path.join(root, f'llmdata/gemma-3-4b-it_except_norm.pt')  # 262144
        datapath = os.path.join(root, f'llmdata/no_compress_tr_mlp_mat.pt')  # 262144
        # datapath = os.path.join(root, f'llmdata/no_compress_tr_mlp_full_mat.pt')  # 262144
        # datapath = "wdata/wae_vae_weights_latent.pt"
        # datapath = os.path.join(root, f'llmdata/gemma-3-4b-it_full_mat_up_proj.pt')  # 262144
        # datapath = os.path.join(root, f'llmdata/gemma-3-4b-it_full_mat_mlp.pt')  # 262144
        #

        self.transform = transform
        # self.transform = transforms.Lambda(RandomSwapTransform(p=0.1))
        # self.transform = transforms.Lambda(lambda x: torch.asinh(x))
        data= self.load_data(datapath, dataset=dataset)

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
        if self.transform:
            weight = self.transform(weight)

        weight= weight/self.scale
        sample = {'weight': weight, 'dataset': []}
        return sample
    def load_data(self, file, dataset='joint'):
        data = torch.load(file)
        wl = []
        if dataset=='joint':
            keys = list(data)
            # if 'lm_head' in keys:
            #     keys.remove('lm_head')
            # keys = ['sharegpt_cot', 'gemini_alpaca_sharegpt']
            # keys =keys[:-1]
            for k in keys:
                w = data[k].reshape(1,-1)
                print(w.shape)
                # exit()

                if self.normalize == "z_score":
                    u = torch.mean(w)
                    v = torch.std(w)
                    w = (w - u) / v

                    # u = torch.mean(w, dim=1)
                    # v = torch.std(w, dim=1)
                    # w = (w - u[:, None]) / v[:, None]
                elif self.normalize == "min_max":
                    x_max, _ = torch.max(w, dim=-1)
                    x_min, _ = torch.min(w, dim=-1)
                    xdiff = x_max - x_min
                    w = 2 * (w - x_min[:, None]) / xdiff[:, None] - 1
                w = pad_to_chunk_multiple(w, chunk_size=self.chunk_size)
                w = w.reshape(1, -1)
                w = torch.split(w, split_size_or_sections=self.max_len, dim=-1)
                w = torch.cat(w, dim=0)

                if self.topk is not None:
                    if self.topk > 0:
                        w = w[:self.topk]
                        wl.append(w)
                else:
                    wl.append(w)
            data = torch.cat(wl, dim=0)
        else:
            w = data[dataset]
            w = pad_to_chunk_multiple(w, chunk_size=self.chunk_size)
            w = torch.split(w, split_size_or_sections=self.chunk_size, dim=-1)
            w = torch.cat(w, dim=0)
            if self.normalize == "z_score":
                u = torch.mean(w, dim=1)
                v = torch.std(w, dim=1)
                w = (w - u[:, None]) / v[:, None]
            elif self.normalize == "min_max":
                x_max, _ = torch.max(w, dim=-1)
                x_min, _ = torch.min(w, dim=-1)
                xdiff = x_max - x_min
                w = (w - x_min[:, None]) / xdiff[:, None]
            if self.topk is not None:
                if self.topk > 0:
                    w = w[:self.topk]

            data = w
        return data


