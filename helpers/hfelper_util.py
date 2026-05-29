import os
import pickle

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
# os.environ["TOKENIZERS_PARALLELISM"] = "false"



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



def collect_flat_weights(
    state_dict: Dict[str, torch.Tensor],
    chunk_size: int,
    skip_if_contains: Sequence[str] = (".bias", ".norm"),
) :
    """Flatten all weights, skip unwanted names, pad to `chunk_size`."""
    # out: Dict[str, torch.Tensor] = {}
    weights = []
    for name, w in state_dict.items():
        if any(s in name for s in skip_if_contains):
            continue
        flat = pad_to_chunk_multiple(w.detach().cpu().reshape(1, -1), chunk_size)
        flat = torch.split(flat, split_size_or_sections=chunk_size, dim=-1)
        flat = torch.cat(flat, dim=0)
        weights.append(flat)
    weights = torch.cat(weights, dim=0)
    return weights

from concurrent.futures import ThreadPoolExecutor, as_completed

def save_block(b, path):
    torch.save(b, path)

def save_blocks_multithreaded(data_dir, model_name, bdata, max_workers=4):
    out_dir = os.path.join(data_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Prepare all (tensor, path) pairs
    tasks = [
        ({'chunk': b, 'idx':i}, os.path.join(out_dir, f"block_{i}_.pt"))
        for i, b in enumerate(bdata)
    ]

    # Launch saves in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(save_block, b, path) for b, path in tasks]
        for f in as_completed(futures):
            # Will raise if any save_block raised
            f.result()


import torch
import pandas as pd
import re

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
def reproduce_metadata_from_state_dict(
        state_dict_path: str,
        output_csv_path: str
):
    """
    Load a PyTorch state_dict, extract parameter names along with
    layer_type and description columns, and save as CSV matching the example format.
    """
    # 1. Load the state dict
    state_dict = torch.load(state_dict_path, map_location='cpu')

    records = []
    for full_name, tensor in state_dict.items():
        # Extract description (last token, e.g. 'weight' or 'bias')
        description = full_name.split('.')[-1]
        # Infer layer_type from preceding name tokens
        parts = full_name.split('.')[:-1]
        layer_type = next(
            (p.lower()
             for p in parts
             if re.match(r'^(conv|bn|batchnorm|downsample|fc|linear)\d*$', p, re.IGNORECASE)),
            'other'
        )

        records.append({
            'name': full_name,
            'layer_type': layer_type,
            'description': description
        })

    # 2. Build DataFrame with the correct column order
    df = pd.DataFrame(records, columns=['name', 'layer_type', 'description'])

    # 3. Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Reproduced metadata saved to: {output_csv_path}")


def collect_flat_weights_with_labels(
    state_dict: Dict[str, torch.Tensor],
    chunk_size: int,
    skip_if_contains: Sequence[str] = (".bias", "norm"),
    select_layers=None,
    ntok =1,
    model_name=None
) :
    """Flatten all weights, skip unwanted names, pad to `chunk_size`."""
    # out: Dict[str, torch.Tensor] = {}
    weights = {}
    records = []
    names =[]
    chunk_records = []
    for name, w in state_dict.items():
        if any(s in name for s in skip_if_contains):
            continue
        if select_layers is not None:
            # skip names that contain none of the substrings
            if not any(st in name for st in select_layers):
                continue

        flat = pad_to_chunk_multiple(w.detach().cpu().reshape(1, -1), chunk_size)
        flat = torch.split(flat, split_size_or_sections=chunk_size, dim=-1)
        flat = torch.cat(flat, dim=0)
        # flat = pad_batch_to_multiple(flat, ntok)
        flat = flat.reshape(-1,chunk_size)
        y = list(range(flat.shape[0]))
        lname =[str(name)]*len(y)
        names.extend(lname)
        chunk_records.extend(y)
        weights[name] = flat

    # weights = torch.cat(weights, dim=0)
    labels = {'chunk_idx': chunk_records, 'LayerName': names}

    return weights, labels

# def collect_flat_weights_with_labels(
#     state_dict: Dict[str, torch.Tensor],
#     chunk_size: int,
#     skip_if_contains: Sequence[str] = (".bias", "norm"),
#     ntok =1,
#     model_name=None
# ) :
#     """Flatten all weights, skip unwanted names, pad to `chunk_size`."""
#     # out: Dict[str, torch.Tensor] = {}
#     weights = {}
#     records = []
#     names =[]
#     chunk_records = []
#     for name, w in state_dict.items():
#         if any(s in name for s in skip_if_contains):
#             continue
#         # weight
#         flat = pad_to_chunk_multiple(w.detach().cpu().reshape(1, -1), chunk_size)
#         flat = torch.split(flat, split_size_or_sections=chunk_size, dim=-1)
#         flat = torch.cat(flat, dim=0)
#         # flat = pad_batch_to_multiple(flat, ntok)
#         flat = flat.reshape(-1,chunk_size)
#         y = list(range(flat.shape[0]))
#         lname =[str(name)]*len(y)
#         names.extend(lname)
#         chunk_records.extend(y)
#         weights[name] = flat
#
#     # weights = torch.cat(weights, dim=0)
#     labels = {'chunk_idx': chunk_records, 'LayerName': names}
#
#     return weights, labels