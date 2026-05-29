
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
import warnings  # put this near the top of your file
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
from typing import List, Dict, Tuple, Sequence
import torch
import  random

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
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    shape = x.shape
    # delta1 = max_in - shape[0]
    delta2 = max_in - shape[-1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out

def preprocess_asinh(x, mu=0, sigma=0.2, lam=0.1):
    z = (x - mu) / sigma
    return torch.asinh(z / lam)              # y  (bounded ~ [-asinh, asinh])

def inv_preprocess_asinh(y, mu, sigma, lam=0.1):
    z = lam * torch.sinh(y)
    return z * sigma + mu

class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, zoo_root, zoo_split, length, resolution, to_image, in_channel, topk,
                 transform=None, target_transform=None, layer_info_path=None, model_info_path=None,
                 scale=1.0, normalize=None):
        super(ZooDataset, self).__init__()
        self.zoo_root = zoo_root
        self.zoo_split = zoo_split
        self.length = length
        self.resolution = resolution
        self.to_image = to_image
        self.in_channel = in_channel
        self.transform = transform
        self.normalize = normalize
        self.target_transform = target_transform
        self.topk = topk
        self.scale = scale
        self.model_info_path = model_info_path
        self.layer_info_path = layer_info_path
        self.data_dir = os.path.join(self.zoo_root, self.zoo_split)
        arch_emb_file = os.path.join(self.zoo_root,zoo_split, f'{zoo_split}_config.pt')
        self.arch_emb = torch.load(arch_emb_file, map_location='cpu', weights_only=False)
        data = glob(f"{self.zoo_root}/{self.zoo_split}/*/*.pt")
        if self.topk is not None:
            data = data[:self.topk]
            # print(data)
            # exit()
        self.files_list = data
        print(f'==========dataset==size==={len(self.files_list)}')

        sigma = 0.0001
        max_noise = 3 * sigma
        p_noise = 0.5  # 50% chance to apply noise
        p_swap = 0.5
        self.pflip=0.50
        # self.transform = transform


        self.randtransform = transforms.Lambda(
            lambda x: x.float() + torch.clamp(sigma * torch.randn_like(x), -max_noise, max_noise)
            if random.random() < p_noise else x
        )

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fl = self.files_list[idx]
        # print('==============================================')
        # print(fl)
        # print('==============================================')
        # print( fl.split("__")[-4])
        # print( '------------------------------------')

        # if self.model_info_path is not None:
        chunk_idx = int(fl.split("__")[-2])
        # print(f'chunk_idx={chunk_idx}')
        model_name = fl.split('/')[-2]
        arch_enb = torch.tensor(self.arch_emb[model_name]).float()
        info_file = os.path.join(self.data_dir,model_name,f'{model_name}_.pkl')
        meta_file = os.path.join(self.data_dir,model_name,f'{model_name}_metadata.csv')

        with open(info_file, 'rb') as file:
            layer_chunk = pickle.load(file)
        layer_name = fl.split("__")[-4] #layer_chunk['LayerName'][chunk_idx]
        # print('==============================================')
        # print(fl)
        # print('==============================================')
        # print( fl.split("__")[-4])
        # print( '------------------------------------')
        # print(layer_name)
        # print('==================================================')
        # assert layer_name.removesuffix(".weight")==str(fl.split("__")[-4])

        sigma = layer_chunk['stats'][layer_name]['sigma']
        mu = layer_chunk['stats'][layer_name]['mu']
        x_min = layer_chunk['stats'][layer_name]['x_min']
        x_max = layer_chunk['stats'][layer_name]['x_max']
        # sigma = layer_chunk['sigma']
        # layer_name = layer_name.removesuffix(".weight")
        # print(layer_name)

        df = pd.read_csv(meta_file)



        try:
            # Grab the first matching description
            layer_info = df.loc[df["LayerName"] == layer_name, "Description"].iloc[0]
        except (IndexError, KeyError):
            # Either the layer wasn’t found or the column is missing
            layer_info = ""  # or set a more suitable default for your use-case
            warnings.warn(f"No description found for layer {layer_name}. "
                          "Using an empty string as a placeholder.")



        weight = torch.load(fl , map_location='cpu',weights_only=False).reshape(-1)
        weight = torch.tensor(weight.detach().cpu().contiguous().float().numpy())
        if self.transform=="arsh":
            weight = preprocess_asinh(weight.float(), mu=0, sigma=0.1, lam=0.1)
        if self.target_transform is not None:
            layer_info = self.target_transform(layer_info)
        weight = self.randtransform(weight)
        # if torch.rand(1).item() < self.pflip:
        #     weight = weight.flip(-1)
        if self.normalize=='z_score':
            weight = (weight-mu)/sigma
        elif self.normalize=='min_max':
            weight = 2*(weight-x_min)/(x_max-x_min)-1
        elif self.normalize=='chunk_min_max':
            weight =2*(weight-weight.min())/(weight.max()-weight.min())-1
        elif self.normalize=='chunk_wise_zscore':
            μ = weight.mean()
            σ = weight.std() + 1e-6
            weight = (weight - μ) / σ
        if self.to_image:
            weight = weight.contiguous().reshape(-1, self.in_channel, self.resolution, self.resolution)
        weight = weight/self.scale

        chunk_idx = torch.tensor(chunk_idx, dtype=torch.long)
        arch_enb = arch_enb.float()
        # print(weight.shape)

        sample = {'weight': weight.float(), 'dataset': { 'arch_emb': arch_enb, 'chunk_idx': chunk_idx,
                  'layer_info': layer_info,
                 }
                  }

        return sample



