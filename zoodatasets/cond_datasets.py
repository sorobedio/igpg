
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

def extract_layers_with_b(std, tgt=None, exclude='norm', chunk_size=512):

    ws = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            if exclude is not None and exclude in params:
                continue

            if tgt is not None and tgt in params:
                if 'bias' in params:
                    continue
                w = std[params].reshape(1, -1)
                p = params.replace('weight', 'bias')
                try:
                    b = std[p].reshape(1, -1)
                    print(f'paramertes============={params}-------{w.shape}------{b.shape}--------')
                    w = torch.cat((w, b), dim=-1)
                except:
                    print(f'paramertes============={params}-------{w.shape}------no bias--------')

                # print(w.shape)
                # print(w.min(), w.max())
                w = pad_to_chunk_multiple(w, chunk_size)
                w =  torch.split(w, chunk_size, dim=-1)
                w = torch.cat(w, dim=0)
                ws.append(w)
            elif tgt is None:
                if 'bias' in params:
                    continue
                w = std[params].reshape(1, -1)
                p = params.replace('weight', 'bias')
                try:
                    b = std[p].reshape(1, -1)
                    print(f'paramertes============={params}-------{w.shape}------{b.shape}--------')
                    w = torch.cat((w, b), dim=-1)
                except:
                    print(f'paramertes============={params}-------{w.shape}------no bias--------')

                # print(w.shape)
                # print(w.min(), w.max())
                w = pad_to_chunk_multiple(w, chunk_size)
                w =  torch.split(w, chunk_size, dim=-1)
                w = torch.cat(w, dim=0)
                ws.append(w)
    ws = torch.cat(ws, dim=0)


    return ws

class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, root='modelzoos', dataset="Llama-3.2-3B-Instruct", split='train', topk=None,
                 scale=1.0, transform=None, normalize=False, tgt=None, exd=None, channel=1,input_size=32,
                 length=3072):
        super(ZooDataset, self).__init__()
        #1960513
        self.topk = topk

        self.split = split
        self.dataset = dataset
        self.normalize = normalize
        self.length = length
        self.tgt = tgt
        self.exd = exd
        self.scale=scale
        self.channel = channel
        self.input_size = input_size

        sigma = 0.0001
        max_noise = 3 * sigma
        p_noise = 0.5  # 50% chance to apply noise
        p_swap = 0.5

        # self.transform = transforms.Lambda(
        #     lambda x: x + torch.clamp(sigma * torch.randn_like(x), -max_noise, max_noise)
        #     if random.random() < p_noise else x
        # )

        # swap_transform = RandomSwapTransform(p=p_swap)
        noise_transform = transforms.Lambda(
            lambda x: x + torch.clamp(sigma * torch.randn_like(x), -max_noise, max_noise)
            if random.random() < p_noise else x
        )

        # --- compose the full pipeline ----------------------------------------------
        self.transform = transforms.Compose([
            transforms.RandomApply([noise_transform], p=p_noise),  # apply noise w.p. 0.5
            # swap_transform  # then (maybe) swap
        ])

        # self.transform = transforms.Lambda(RandomSwapTransform(p=0.5))
        # self.transform = transforms.Lambda(lambda x: torch.asinh(x))

        datapath = os.path.join(root, f'zoo_config.simple_config.yaml')  # 262144

        data, self.target = self.load_data(datapath, dataset=dataset)

        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}===m=======')
        # # data = 2 * (data - x_min) / (x_max - x_min) - 1
        mu = data.mean()
        std = data.std()
        # data = (data-mu)/std
        print(f'============{std}==============={mu}======labels:  {self.target.max()}=======')
        # data = data.repeat(10, 1)
        self.data = data.detach().cpu()
        print(f'=={len(self.target)}==={self.target.shape}==dataset size=={data.shape}===max={data.max()}==={data.min()}==========')

    def __len__(self):
        if self.topk is not None:
            return self.topk
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx]
        target = self.target[idx]
        if self.transform:
            weight = self.transform(weight)

        weight= weight/self.scale
        sample = {'weight': weight, 'dataset': target}
        return sample
    def load_data(self, file, dataset="Llama-3.2-3B-Instruct"):
        models_list = load_models(file, dataset)
        print(len(models_list))

        data = []
        labels =[]
        jj =0
        for model in models_list:
            if "gemma-3" in model:

                model = Gemma3ForCausalLM.from_pretrained(
                    model,
                    device_map="cpu",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    # revision="step143000",
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    device_map="cpu",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    # revision="step143000",
                )
            # model = Gemma3ForCausalLM.from_pretrained(dataset, torch_dtype=torch.bfloat16).to('cpu')
            std = model.state_dict()
            del model

            w = extract_layers_with_b(std, tgt=self.tgt, exclude=self.exd, chunk_size=self.length)

            y = torch.tensor(list(range(jj, w.shape[0] + jj)))
            labels.extend(y.reshape(-1).tolist())
            jj += w.shape[0]
            data.append(w)
        data = torch.cat(data, dim=0)
        data = data.reshape(-1, self.channel, self.input_size, self.input_size)
        labels = torch.tensor(labels,dtype=torch.long)
        return data, labels