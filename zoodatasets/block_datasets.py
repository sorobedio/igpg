
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
import random

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


def select_random_consecutive_rows(tensor: torch.Tensor, n: int) -> torch.Tensor:
    """
    Selects a random block of n consecutive rows from a 2D tensor.

    Args:
        tensor (torch.Tensor): The input 2D tensor (e.g., shape [num_rows, num_cols]).
        n (int): The number of consecutive rows to select.

    Returns:
        torch.Tensor: A tensor containing the n selected consecutive rows.

    Raises:
        ValueError: If the tensor is not 2D or if n is larger than the number of rows.
    """
    if tensor.ndim != 2:
        raise ValueError(f"Input tensor must be 2D, but got {tensor.ndim} dimensions.")

    num_rows = tensor.shape[0]

    if n < 0:
        raise ValueError("Number of rows 'n' cannot be negative.")
    if n > num_rows:
        raise ValueError(f"Cannot select n={n} rows from a tensor with only {num_rows} rows.")

    # The last possible starting row index is (num_rows - n)
    max_start_index = num_rows - n

    # Randomly choose a starting index for the block
    start_index = random.randint(0, max_start_index)

    # Define the end index of the slice
    end_index = start_index + n

    # Slice the tensor to get the block of consecutive rows
    # The ":" means we take all columns for the selected rows.
    return tensor[start_index:end_index, :]



class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, zoo_root, zoo_split, length, resolution, to_image, in_channel, topk,
                 transform=None, scale=1.0):
        super(ZooDataset, self).__init__()
        self.zoo_root = zoo_root
        self.zoo_split = zoo_split
        self.length = length
        self.resolution = resolution
        self.to_image = to_image
        self.in_channel = in_channel
        self.transform = transform
        self.topk = topk
        self.scale = scale

        self.data_dir = os.path.join(self.zoo_root, self.zoo_split)
        data = glob(f"{self.zoo_root}/{self.zoo_split}/*/*.pt")
        # data = glob(f"{self.zoo_root}/{self.zoo_split}")
        print(f"{len(data)} files found in {self.zoo_root}/{self.zoo_split}")
        if self.topk is not None:
            data = data[:self.topk]
        self.files_list = data


    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fl = self.files_list[idx]
        weight = torch.load(fl , map_location='cpu',weights_only=False)
        idx = torch.randperm(weight.size(0))[0]
        weight = weight[idx]
        if self.transform is not None:
            weight = self.transform(weight)

        weight = torch.tensor(weight.cpu().contiguous().float().numpy()).to(weight.dtype)
        if self.to_image:
            weight = weight.contiguous().reshape(self.in_channel, self.resolution, self.resolution)
        else:
            weight = weight.contiguous().reshape(self.length,)
        weight = weight/self.scale
        sample = {'weight': weight, 'chunk_idx':[], 'layer_info':[]}
        return sample



