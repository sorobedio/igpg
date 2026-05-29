import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import os
import math
import yaml

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

# Example Dataset that holds flattened weight vectors.
class WeightDataset(Dataset):
    def __init__(self, root='zoodata', dataset="joint", split='train',
                 topk=None, scale=1.0, transform=None, normalize=False,
                 max_len=12288):
        """
        Args:
            weight_list (List[torch.Tensor]): A list of 1D tensors (flattened weight vectors).
        """
        self.topk = topk
        self.max_len = max_len
        self.split = split
        self.dataset = dataset
        self.normalize = normalize
        self.chunk_size = max_len
        self.scale = scale
        datapath = os.path.join("../Datasets", f'llmdata/smollm_tf_.pt')
        self.weight_list = self.load_data(datapath, dataset=dataset)


    def __len__(self):
        return len(self.weight_list)

    def __getitem__(self, idx):
        return self.weight_list[idx]

    def load_data(self, file, dataset='joint'):
        data = torch.load(file)
        wl = []
        if dataset=='joint':
            keys = list(data)
            # keys.remove('layernorm.weight')
            # keys = ['sharegpt_cot', 'gemini_alpaca_sharegpt']
            # keys =keys
            # print(keys)
            for k in keys:
                w = data[k].detach().cpu()
                wl.append(w)
        return wl



# Collate function that pads each sample in the batch to the length of the longest sample.
def dynamic_pad_collate(batch):
    """
    Args:
        batch: list of 1D tensors of varying lengths.
    Returns:
        padded: Tensor of shape (B, max_length) with zero-padding.
        lengths: Tensor of original lengths.
    """
    lengths = [x.size(0) for x in batch]
    max_len = max(lengths)
    padded = [F.pad(x, (0, max_len - x.size(0))) for x in batch]
    return torch.stack(padded), torch.tensor(lengths)


# Bucket the dataset so that samples of similar lengths are grouped together.
def create_bucketed_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    # Sort dataset indices by the length of the corresponding sample.
    sorted_indices = sorted(range(len(dataset)), key=lambda i: dataset[i].size(0))

    # Divide sorted indices into batches.
    batches = [sorted_indices[i:i + batch_size] for i in range(0, len(sorted_indices), batch_size)]
    random.shuffle(batches)  # Shuffle the order of batches to add randomness.

    class ListBatchSampler(torch.utils.data.Sampler):
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    batch_sampler = ListBatchSampler(batches)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=dynamic_pad_collate)
    return dataloader

