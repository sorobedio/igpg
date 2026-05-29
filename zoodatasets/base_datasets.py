
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from glob import glob
import math
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
import torch
import  random
from typing import List, Dict, Tuple, Sequence
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    skip_if_contains: Sequence[str] = (".bias", ".norm"),
    select_layers=None
) :
    """Flatten all weights, skip unwanted names, pad to `chunk_size`."""
    # out: Dict[str, torch.Tensor] = {}
    weights = []
    for name, w in state_dict.items():
        if any(s in name for s in skip_if_contains):
            continue
        if select_layers is not None:
            if any(st not in name for st in select_layers):
                continue
        flat = pad_to_chunk_multiple(w.detach().cpu().reshape(1, -1), chunk_size)
        flat = torch.split(flat, split_size_or_sections=chunk_size, dim=-1)
        flat = torch.cat(flat, dim=0)
        weights.append(flat)
    weights = torch.cat(weights, dim=0)
    return weights

class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, zoo_root='modelzoos', zoo_name="Llama-3.2-3B-Instruct", split='train', topk=None,
                 scale=1.0, transform=None, normalize=False, tgt=None, exd=None, toimage_shape=False, sizes=[3, 256],
                 length=3072, zoo_file='zoo_config.simple_config.yaml'):
        super(ZooDataset, self).__init__()
        #1960513
        self.topk = topk
        self.split = split
        self.zoo_name = zoo_name
        self.zoo_root = zoo_root
        self.normalize = normalize
        self.length = length
        self.tgt = tgt
        self.exd = exd
        self.scale=scale
        self.filename = zoo_file
        # self.resize = resize
        self.toimage_shape = toimage_shape
        self.sizes = sizes
        self.transform = transform

        datapath = os.path.join(zoo_root, f'zoo_config.simple_config.yaml')  # 262144

        data= self.load_data(datapath, dataset=zoo_name)

        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')
        # x_min = data.min()
        # x_max = data.max()
        # data = 2 * (data - x_min) / (x_max - x_min) - 1
        mu = data.mean()
        std = data.std()
        # data = (data-mu)/std
        print(f'============{std}==============={mu}=============')
        # data = data.repeat(10, 1)
        self.data = data.detach().cpu().to(torch.bfloat16)
        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx]
        if self.transform is not None:
            weight = self.transform(weight)

        weight= weight/self.scale
        sample = {'weight': weight, 'dataset': []}
        return sample
    def load_data(self, file, dataset="Llama-3.2-3B-Instruct"):
        models_list = load_models(file, dataset)
        print(len(models_list))

        data = []
        wl =[]
        for model_id in models_list:
            print(f'=PROCESSING---model--{model_id}=====:')
            if "gemma" in models_list:

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
            del model
            w = collect_flat_weights(std, chunk_size=self.length, skip_if_contains=self.exd,
                                     select_layers=self.tgt)
            if self.toimage_shape and self.sizes is not None:
                w = w.reshape(-1, self.sizes[0], self.sizes[1], self.sizes[1])
            data.append(w)
            print(f'=model--{model_id}====has==:{w.shape}')

        data = torch.cat(data, dim=0)
        if self.topk is not None:
            data = data[:self.topk]

        return data.float()