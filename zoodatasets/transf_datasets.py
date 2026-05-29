
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from glob import glob
import math
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
from typing import List, Dict, Tuple, Sequence
import torch
import  random

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


def load_models(yaml_file, model_name=None):
    """
    Load the YAML file and extract models.

    Parameters:
        yaml_file (str): Path to the YAML file.
        model_name (str, optional): If provided, returns only the list for this key.

    Returns:
        list: If model_name is provided, returns the list of model pointers for that key.
              If model_name is None, returns a flattened list of all model pointers.
              Returns None if the file cannot be read or if the key is not found.
    """
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
    except Exception as e:
        print("Error reading YAML file:", e)
        return None

    if model_name:
        models_list = data.get(model_name)
        if models_list is None:
            print(f"Model '{model_name}' not found in the YAML file.")
        return models_list
    else:
        # Flatten all model pointers into a single list.
        all_models = []
        for key, models in data.items():
            if isinstance(models, list):
                all_models.extend(models)
            else:
                all_models.append(models)
        return all_models




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

def pad_batch_to_multiple(x: torch.Tensor, chunk_size: int):
    """
    Zero-pads the batch dimension (dim 0) so its size becomes a multiple of `chunk_size`.

    Returns
    -------
    padded_x : torch.Tensor
        Tensor with batch size divisible by `chunk_size`.
    pad_len : int
        Number of added dummy examples.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)                       # treat 1-D vector as single-item batch

    pad_len = (-x.shape[0]) % chunk_size         # how many to add (mod trick)
    if pad_len:
        pad = x.new_zeros(pad_len, *x.shape[1:]) # same dtype/device
        x = torch.cat((x, pad), dim=0)

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

def get_weights_mat(std,xcond, typ=None):
    weights ={}
    if typ is None:
        for k in std:
            if xcond in k:
                continue
            w = std[k].detach().cpu()
            weights[k] = w
            print(f'param:{k}--shape:{w.shape}--min:{w.min()}--max:{w.max()}')
            print('-----------------------------------------------------------')

    else:
        for k in std:
            if xcond in k:
                continue
            if typ in k:
                w = std[k].detach().cpu()
                weights[k] = w
                print(f'param:{k}--shape:{w.shape}--min:{w.min()}--max:{w.max()}')
                print('-----------------------------------------------------------')
    return weights


def collect_flat_weights(
    state_dict: Dict[str, torch.Tensor],
    chunk_size: int,
    skip_if_contains: Sequence[str] = (".bias", "norm"),
    select_layers=None
) :
    """Flatten all weights, skip unwanted names, pad to `chunk_size`."""
    # out: Dict[str, torch.Tensor] = {}
    weights = []
    for name, w in state_dict.items():
        if any(s in name for s in skip_if_contains):
            continue
        if select_layers is not None:
            # skip names that contain none of the substrings
            if not any(st in name for st in select_layers):
                continue
        # if select_layers is not None:
        #     if any(st not in name for st in select_layers):
        #         continue

        flat = pad_to_chunk_multiple(w.detach().cpu().reshape(1, -1), chunk_size)
        flat = torch.split(flat, split_size_or_sections=chunk_size, dim=-1)
        flat = torch.cat(flat, dim=0)
        flat =  torch.tensor(flat.detach().cpu().float().numpy()).to(flat.dtype)
        print(f'param:{name}--shape:{flat.shape}')
        weights.append(flat)
    weights = torch.cat(weights, dim=0)
    return weights

def preprocess_asinh(x, mu, sigma, lam=0.1):
    z = (x - mu) / sigma
    return torch.asinh(z / lam)              # y  (bounded ~ [-asinh, asinh])

def inv_preprocess_asinh(y, mu, sigma, lam=0.1):
    z = lam * torch.sinh(y)
    return z * sigma + mu

class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, root='modelzoos', split="Llama-3.2-3B-Instruct", topk=None,
                 scale=1.0, transform=None, normalize=False, tgt=None, exd=None, to_image=False, in_ch=3,
                 length=3072, zoo_file=None, n_tok=64, input_size=224, lamda=0.1):
        super(ZooDataset, self).__init__()
        #1960513
        self.topk = topk

        self.split = split

        self.normalize = normalize
        self.length = length
        self.tgt = tgt
        self.exd = exd
        self.scale=scale
        self.n_tok = n_tok
        self.input_size = input_size
        self.zoo_file = zoo_file
        self.to_image = to_image
        self.in_ch = in_ch
        # self.scale=scale
        # x_max = 14.375
        # x_min = -14.125

        sigma = 0.001
        max_noise = 3 * sigma
        p_noise = 0.25  # 50% chance to apply noise
        p_swap = 0.25
        self.transform = transform

        # self.transform = transforms.Lambda(
        #     lambda x: x + torch.clamp(sigma * torch.randn_like(x), -max_noise, max_noise)
        #     if random.random() < p_noise else x
        # )

        swap_transform = RandomSwapTransform(p=p_swap)
        noise_transform = transforms.Lambda(
            lambda x: x + torch.clamp(sigma * torch.randn_like(x), -max_noise, max_noise)
            if random.random() < p_noise else x
        )
        #
        # # --- compose the full pipeline ----------------------------------------------
        self.transform = transforms.Compose([
            transforms.RandomApply([noise_transform], p=p_noise),  # apply noise w.p. 0.5
            swap_transform  # then (maybe) swap
        ])

        # self.transform = transforms.Lambda(RandomSwapTransform(p=0.5))
        # self.transform = transforms.Lambda(lambda x: torch.asinh(x))

        datapath = os.path.join(root, f'zoo_config.yaml')  # 262144

        data, self.targets = self.load_data(datapath, dataset=split)

        # print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')
        # dtype = data.dtype
        # data = 2 * (data.float() - x_min) / (x_max - x_min) - 1
        # mu = data.mean()
        # std = data.std()
        # data = (data-mu)/std
        # print(f'============{std}==============={mu}=============')
        # data = data.repeat(10, 1)
        self.data = data
        print(f'===============dataset size=={len(data)}==========')
        self.mu = 3.4088494430761784e-05
        self.std = 0.07324592024087906
        self.lamda = lamda

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            weight = self.transform(weight)
        # weight = preprocess_asinh(weight, self.mu, self.std, self.lamda)
        if torch.rand(1) < 0.25:
            weight = torch.flip(weight, dims=[-1])

        weight= weight/self.scale
        sample = {'weight': weight, 'dataset': target}
        return sample
    def load_data(self, file, dataset="Llama-3.2-3B-Instruct"):
        models_list = load_models(file, dataset)
        print(len(models_list))

        data = []
        ylabels =[]
        jj =0
        for model_id in models_list:
            if "gemma-3" in model_id:

                model = Gemma3ForCausalLM.from_pretrained(
                    model_id,
                    device_map="cpu",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    # revision="step143000",
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="cpu",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    # revision="step143000",
                )
            # model = Gemma3ForCausalLM.from_pretrained(dataset, torch_dtype=torch.bfloat16).to('cpu')
            std = model.state_dict()

            wl =  collect_flat_weights(std, chunk_size=self.length, skip_if_contains=self.exd,
                                     select_layers=self.tgt)
            wl = pad_batch_to_multiple(wl, chunk_size=self.n_tok)
            wl = wl.reshape(-1,self.n_tok*self.length)
            if self.to_image:
                wl =wl.reshape(-1, self.in_ch,self.input_size,self.input_size)
            data.append(wl)
            ylabels.append(jj)
            jj +=1
            # print(f"============model=={model_id}==={wl.shape}")
            del model

        # data = torch.cat(data, dim=0)
        if self.topk is not None:
            data = data[:self.topk]
        ylabels = torch.tensor(ylabels,dtype=torch.long).reshape(-1,)
        return data, ylabels