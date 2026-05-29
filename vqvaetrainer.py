"""
Distributed AutoEncoder Training Script (Single‑Node Multi‑GPU)
==============================================================
This is a **drop‑in replacement** for your previous script, rewritten to use
`torch.distributed` with `DistributedDataParallel` (DDP).  Launch with e.g.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 distributed_autoencoder.py \
        --data modelzoos --dataset joint --split train --n_epochs 1000
```
Key changes
~~~~~~~~~~~
1. Initialise the process‑group from the environment variables that `torchrun`
   sets (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`).
2. One GPU **per** process (`torch.cuda.set_device(local_rank)`).
3. Use `DistributedSampler` for the dataset and call `sampler.set_epoch(epoch)`.
4. Wrap the model with `torch.nn.parallel.DistributedDataParallel`.
5. Log / save checkpoints **only** on rank‑0.
6. Clean shutdown of the process‑group (`dist.destroy_process_group()`).

Everything else—optimisers, mixed‑precision, custom training loop—remains
unchanged.
"""

# ─────────────────────────────────────────────────────────────
# Imports & Setup
# ─────────────────────────────────────────────────────────────
import argparse, os, sys, datetime, random, yaml, math
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from packaging import version
from omegaconf import OmegaConf
from tqdm import tqdm

import torchvision
from PIL import Image
import pytorch_lightning as pl  # (kept for callbacks/helpers)

from helpers.helpers import *
from helpers.misc import progress_bar
from utils.util import instantiate_from_config
from zoodatasets.tf_datasets import ZooDataset
# from zoodatasets.autoloader import ZooDataset
import math
from torch.optim.lr_scheduler import _LRScheduler

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models

# Optional logging libs ---------------------------------------------------------
try:  # wandb may not be installed on all ranks
    import wandb
except ImportError:
    wandb = None


# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# ─────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────

def seed_everything(seed: int = 25):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 2. Wrap in a small head that pools & flattens to 256‑D
class ResNet18_256(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 256, 1, 1)
    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)["feat"]         # (B, 256, H, W)
        # x = self.pool(x).flatten(1)          # -> (B, 256)
        return x
class WarmUpCosineScheduler(_LRScheduler):
    """
    PyTorch LR scheduler with warmup and cosine decay.
    - Set base_lr in optimizer.
    - Steps from lr_start to lr_max during warmup.
    - Cosine anneal from lr_max to lr_min after warmup.
    """

    def __init__(
        self,
        optimizer,
        warm_up_steps,
        lr_min,
        lr_max,
        lr_start,
        max_decay_steps,
        last_epoch=-1,
        verbosity_interval=0,
    ):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.verbosity_interval = verbosity_interval
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        n = self.last_epoch

        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
        else:
            t = (n - self.lr_warm_up_steps) / max(1, self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(t * math.pi))

        if self.verbosity_interval > 0 and n % self.verbosity_interval == 0:
            print(f"current step: {n}, lr: {lr}")

        # If there are multiple param groups, scale for each (default: same for all)
        return [lr for _ in self.optimizer.param_groups]

# ─────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────

def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")
#"modelzoos"

    p = argparse.ArgumentParser(description="Distributed AutoEncoder Training")
    p.add_argument("--data", default="modelzoos")
    p.add_argument("--data_root", default="../Datasets/")
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--dataset", default="joint")
    p.add_argument("--split", default="train")
    p.add_argument("--ae_type", default="ldm")
    p.add_argument("--save_path", default="autocheckpoints")
    p.add_argument("--n_epochs", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--name", type=str, default="adt")
    p.add_argument("--base",
                   default="vqvae_igpg/configs/vqvae_tiny+subset_64.yaml",
                   # default="vqvae_igpg/configs/train_set_config_.yaml"
                   # default="vqvae_igpg/configs/llm_vitvqvae_config.yaml"
                   # default="vqvae_igpg/configs/small_1b_model_base_config.simple_config.yaml"
    # default = "vqvae_igpg/configs/first_stage_config_vqvae.simple_config.yaml"
                   #
                   )
    p.add_argument("--resume", default="")
    p.add_argument("--wandb", type=str2bool, nargs="?", const=True, default=True)
    return p


# ─────────────────────────────────────────────────────────────
# DDP Initialisation Helpers
# ─────────────────────────────────────────────────────────────dev
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:  # not launched with torchrun → fallback to single‑GPU
        rank = 0
        world_size = 1
        local_rank = 0
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def is_main_process(rank):
    return rank == 0

def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use
# ─────────────────────────────────────────────────────────────
# Training / Evaluation Loops
# ─────────────────────────────────────────────────────────────
def train(model, optimizer, scaler, train_loader, sampler, rank, n_epochs, save_path):
    if is_main_process(rank):
        os.makedirs(save_path, exist_ok=True)
    loss_fn = model.module.loss
    global_step = 0
    best_epoch = 0

    best_loss = math.inf
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)  # shuffles shards differently each epoch
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"[Rank {rank}] Epoch {epoch+1}") if is_main_process(rank) else enumerate(train_loader)

        for step, batchs in pbar:

            cond = batchs["dataset"]
            batch = batchs['weight'].squeeze(1).to(device)

            optimizer.zero_grad(set_to_none=True)
            #: { 'arch_emb': arch_enb, 'chunk_idx': chunk_idx,
            # 'layer_info': layer_info,
            # }
            with torch.autocast("cuda", dtype=torch.float32):
                batch, xrec, qloss, indices = model(batch, return_indices=True)
            loss, logs = loss_fn(qloss, batch.float(), xrec.float(), split="train",
                                 predicted_indices=indices)

            # logs['train/temp_t']= model.module.temperature_scheduling(global_step)
            global_step += 1
            # imgw = batch.reshape(batch.shape[0], 3, 192, 192).float()
            # x_feat = feature_extractor(imgw)
            # imgw_rec = xrec.reshape(xrec.shape[0], 3, 192, 192).float()
            # rec_feat = feature_extractor(imgw_rec)
            # bloss = F.mse_loss(x_feat, rec_feat)
            # loss = loss + bloss
            b = batch.size(0)
            # uloss =  (xrec.reshape(b, -1).float().mean(dim=1, keepdim=True)-
            #           batch.reshape(b, -1).float().mean(dim=1, keepdim=True))**2
            sigma_loss = (xrec.reshape(b, -1).float().std(dim=1, keepdim=True) -
                          batch.reshape(b, -1).float().std(dim=1, keepdim=True))**2

            loss = loss  + sigma_loss.mean()
                # loss, logs = model.training_step(batch, step)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if is_main_process(rank):
                pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

                # per‑step logging (rank‑0 only) ---------------------------------
            if is_main_process(rank) and wandb and wandb.run is not None:
                rec_loss = logs['train/rec_loss']
                kld_loss = logs['train/quant_loss']
                nnl_loss = logs['train/total_loss']
                # temp_t = logs['train/temp_t']
                perplexity =logs[f"train/perplexity"]
                cluster_usage =logs[f"train/cluster_usage"]
                wandb.log({
                    "epoch": epoch + 1,
                    # "bh_loss": bloss.item(),
                    "rec_loss": rec_loss.item(),
                    "qunat_loss": kld_loss.item(),
                    "nll_loss": nnl_loss.item(),
                    # "temp_t": temp_t,
                    "cluster_usage": cluster_usage,
                    "perplexity": perplexity,
                })



        # Reduce the epoch loss across ranks
        loss_tensor = torch.tensor(epoch_loss / len(train_loader), device="cuda")
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        reduced_loss = loss_tensor.item() / dist.get_world_size()

        if is_main_process(rank):
            print(f"Epoch {epoch+1} | loss={reduced_loss:.4f}")

            # checkpoint
            if reduced_loss < best_loss:
                best_loss = reduced_loss
                best_epoch = epoch + 1
                torch.save(model.module.state_dict(), os.path.join(save_path, "model_gemma_vqvae_normal.pt"))
                # torch.save(model, os.path.join(save_path, "model_vqvae_best_model_128_ln.pth"))
            print(f"best-epoch-loss--Epoch {best_epoch} | loss={best_loss:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train_epoch_loss": reduced_loss,
            })
        if is_main_process(rank):
            if (epoch + 1) % 10 == 0:

                print(f'Input: {batch.reshape(batch.shape[0], -1)[0, :10].detach().cpu()},'
                      f' Dec: {xrec.reshape(xrec.shape[0], -1)[0, :10].detach().cpu()}')
                print(f'Input-min: {batch.detach().cpu().min()} Input-ax: {batch.detach().cpu().max()},'
                           f' Dec-min: {xrec.detach().cpu().min()}  Dec-min: {xrec.detach().cpu().max()}')


        # # log_var = logs['train/logvar']
        # if args.wandb and is_main_process(rank)





def m_collate(batch):
    sample = {}

    data = [item['weight'] for item in batch]

    data = torch.stack(data, 0)
    return data
# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args, unknown = get_parser().parse_known_args()

    # Distributed init ----------------------------------------------------------------
    rank, world_size, local_rank = init_distributed_mode()

    # Reproducibility ------------------------------------------------------------------
    seed_everything(args.seed + rank)  # different seed per rank

    # Config / model -------------------------------------------------------------------
    configs = [OmegaConf.load(args.base)]
    cli_cfg = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(*configs, cli_cfg)
    model = instantiate_from_config(cfg.model)
    params = count_trainable_parameters(model)
    base_lr = model.learning_rate

    resnet = models.resnet18(weights="IMAGENET1K_V1")
    # return the activation **after** layer3’s last ReLU
    return_nodes = {"layer3.1.relu": "feat"}
    feature_extractor_base = create_feature_extractor(resnet, return_nodes).cuda()

    feature_extractor = ResNet18_256(feature_extractor_base).eval()
    for p in feature_extractor.parameters():
        p.requires_grad = False


    feature_extractor.eval()

    model.cuda()

    # Wrap in DDP (find_unused_parameters=False unless you need it)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # Optimiser & AMP ------------------------------------------------------------------
    optimizer = model.module.configure_optimizers()
    scheduler = WarmUpCosineScheduler(
        optimizer,
        warm_up_steps=0,
        lr_min=1e-7,
        lr_max=1e-4,
        lr_start=base_lr,
        max_decay_steps=1000001,
        verbosity_interval=0,
    )
    # scaler = torch.GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    # Dataset / Loader -----------------------------------------------------------------
    train_set = ZooDataset(
        root=args.data,
        dataset="gemma-3-4b-it",
        split=args.split,
        scale=1,
        tgt=["k_proj"],
        exd=["norm", "bias"],
        length=16384,
        to_image=False,
        n_tok=1,
        input_size=128,
        in_ch=3
    )
    # zoo_root:  '../zoodatasets_train_tiny'
    # zoo_split: "train_tiny"
    # length: 110592
    # resolution: 192
    # train_set= ZooDataset(zoo_root ='../zoodatasets_train_tiny_subset-64',
    #                       zoo_split = "train_tiny_subset-2",
    #                       length = 12288,
    #                       resolution = 64,
    #                       to_image = True,
    #                       in_channel = 3,
    #                       normalize='chunk_wise_zscore',
    #                       topk = None,
    #                       scale = 1.0,
    #                       transform=None,
    #                         )

    # data_dir = '../zoodatasets_train_tiny_subset_128'
    # file_name = 'modelzoos/zoo_config.simple_config.yaml'
    # target_set = "train_tiny_subset"
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        sampler=sampler,
        # collate_fn=m_collate,
    )

    # WandB (only initialise on rank‑0) -----------------------------------------------
    if args.wandb and is_main_process(rank) and wandb is not None:
        wandb.init(
            project="gemma_vqvae_project",
            #
            # project="ddp_ae_project",
            name=f"{args.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=dict(
                epochs=args.n_epochs,
                batch_size=args.batch_size,
                lr=optimizer.param_groups[0]['lr'],
            ),
            mode="online",
        )
    print(f'=========model===with==:{params//1e6} M========parameters')
    # Training ------------------------------------------------------------------------
    train(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_loader=train_loader,
        sampler=sampler,
        rank=rank,
        n_epochs=args.n_epochs,
        save_path=args.save_path,
    )

    # Clean‑up -------------------------------------------------------------------------
    dist.barrier()
    dist.destroy_process_group()

#torchrun --standalone --nnodes=1 --nproc_per_node=7 vqvaetrainer.py \
        # --data modelzoos --dataset joint --split train --n_epochs 1000
        #torchrun --standalone --nnodes=1 --nproc_per_node=1 vqvaetrainer.py
        #torchrun --standalone --nnodes=1 --nproc_per_node=8 vqvaetrainer.py