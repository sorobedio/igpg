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
import torch
import torch.nn.functional as F
from torch.distributions import Normal, StudentT

import torchvision
from PIL import Image
import pytorch_lightning as pl  # (kept for callbacks/helpers)

from helpers.helpers import *
from helpers.misc import progress_bar
from utils.util import instantiate_from_config
# from zoodatasets.tf_datasets import ZooDataset
from zoodatasets.basedatasets import ZooDataset
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.distributions import Normal, StudentT, Laplace

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models
import torch
# dummy = torch.randn(4, 3, 192, 192)



# Optional logging libs ---------------------------------------------------------
try:  # wandb may not be installed on all ranks
    import wandb
except ImportError:
    wandb = None


os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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

    p = argparse.ArgumentParser(description="Distributed AutoEncoder Training")
    p.add_argument("--data", default="modelzoos")
    p.add_argument("--data_root", default="../Datasets/")
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--dataset", default="joint")
    p.add_argument("--split", default="train")
    p.add_argument("--ae_type", default="ldm")
    p.add_argument("--save_path", default="tiny_autocheckpoints")
    p.add_argument("--n_epochs", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=300)
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--name", type=str, default="adt")
    p.add_argument("--base",
                   # default="vit_vae/configs/base_autoencoder_kl.simple_config.yaml"
                   default="vit_vae/configs/base_4_ratio_config_kl.yaml"
                   # default="vqvae_igpg/configs/first_stage_config_vqvae.simple_config.yaml"
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

# # 2. Wrap in a small head that pools & flattens to 256‑D
# class ResNet18_256(nn.Module):
#     def __init__(self, backbone):
#         super().__init__()
#         self.backbone = backbone
#         self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 256, 1, 1)
#     def forward(self, x):
#         x = self.backbone(x)["feat"]         # (B, 256, H, W)
#         # x = self.pool(x).flatten(1)          # -> (B, 256)
#         return x



def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) loss between tensors x and y.

    Args:
        x: tensor of shape (batch_size, feature_dim)
        y: tensor of shape (batch_size, feature_dim)
        sigma: Gaussian kernel bandwidth. If your data is normalized (std=1),
               sigma=1.0 is a good default.

    Returns:
        scalar MMD loss.
    """

    def gaussian_kernel(a, b):
        a = a.unsqueeze(1)  # (batch_size, 1, feature_dim)
        b = b.unsqueeze(0)  # (1, batch_size, feature_dim)
        dist_sq = ((a - b) ** 2).sum(dim=2)  # (batch_size, batch_size)
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    k_xx = gaussian_kernel(x, x).mean()
    k_yy = gaussian_kernel(y, y).mean()
    k_xy = gaussian_kernel(x, y).mean()

    return k_xx + k_yy - 2 * k_xy


def vae_loss(recon, x, mu, logvar, kl_weight=0.0001):
    """
    Computes the loss for a variational autoencoder.

    Args:
        recon (Tensor): Reconstructed tensor.
        x (Tensor): Input tensor.
        mu (Tensor): Latent mean.
        logvar (Tensor): Latent log-variance.

    Returns:
        loss (Tensor): Total loss (reconstruction + KL divergence).
        recon_loss (Tensor): Reconstruction loss.
        kl_loss (Tensor): KL divergence loss.
    """

    # mask = (x != 0).float()

    # imgw = x.reshape(x.shape[0], 3, 256, 256)
    # x_feat = feature_extractor(imgw)
    # imgw_rec = recon.reshape(recon.shape[0], 3, 256, 256)
    # rec_feat = feature_extractor(imgw_rec)
    # bloss =F.mse_loss(x_feat, rec_feat)
    #
    # ---------- 1. Reconstruction term ----------
    # recon_flat = _per_sample_flat(recon)
    x_flat = x.reshape(x.shape[0], -1)
    recon_flat = recon.reshape(recon.shape[0], -1)

    # compute μ and σ along the *last* (feature) dimension, keepdim for broadcasting
    rec_mean = recon_flat.mean(dim=-1, keepdim=True)
    rec_std = recon_flat.std(dim=-1, keepdim=True) + 1e-8

    x_mean = x_flat.mean(dim=-1, keepdim=True)
    x_std = x_flat.std(dim=-1, keepdim=True) + 1e-8

    x_rec_norm = (recon_flat - rec_mean) / rec_std
    x_norm = (x_flat - x_mean) / x_std

    recon_loss = (
            F.mse_loss(recon, x, reduction="mean")*1000 +
            F.mse_loss(x_rec_norm, x_norm, reduction="mean")
            # bloss*10
    )

    # KL divergence loss (averaged over the batch)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss * kl_weight


def _per_sample_flat(t: torch.Tensor) -> torch.Tensor:
    """
    (B, C, …) → (B, F)  where F = C×…
    Keeps batch dim, flattens the rest.
    """
    return t.view(t.size(0), -1)

#
# def vae_loss(
#     recon: torch.Tensor,
#     x: torch.Tensor,
#     mu: torch.Tensor,
#     logvar: torch.Tensor,
#     *,
#     prior: str = "gaussian",
#     df: float = 5.0,
#     kl_weight: float = 1e-4,
# ):
#     # ---------- 1. Reconstruction term ----------
#     recon_flat = _per_sample_flat(recon)
#     x_flat     = _per_sample_flat(x)
#
#     # compute μ and σ along the *last* (feature) dimension, keepdim for broadcasting
#     rec_mean = recon_flat.mean(dim=-1, keepdim=True)
#     rec_std  = recon_flat.std(dim=-1, keepdim=True) + 1e-8
#
#     x_mean   = x_flat.mean(dim=-1, keepdim=True)
#     x_std    = x_flat.std(dim=-1, keepdim=True) + 1e-8
#
#     x_rec_norm = (recon_flat - rec_mean) / rec_std
#     x_norm     = (x_flat     - x_mean)   / x_std
#
#     recon_loss = (
#         F.mse_loss(recon_flat, x_flat, reduction="mean") +
#         F.mse_loss(x_rec_norm, x_norm, reduction="mean")
#     )
#
#     # ---------- 2. KL divergence (unchanged) ----------
#     if prior == "gaussian":
#         kl = 0.5 * torch.sum(
#             torch.exp(logvar) + mu**2 - 1.0 - logvar,
#             dim=1
#         )
#     elif prior == "student":
#         q = Normal(mu, torch.exp(0.5 * logvar))
#         with torch.no_grad():
#             z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
#         log_qzx = q.log_prob(z).sum(-1)
#         p = StudentT(df=df, loc=torch.zeros_like(z), scale=torch.ones_like(z))
#         log_pz = p.log_prob(z).sum(-1)
#         kl = log_qzx - log_pz
#     else:
#         raise ValueError("prior must be 'gaussian' or 'student'")
#
#     kl_loss = kl.mean()
#     total   = recon_loss + kl_weight * kl_loss
#     return total, recon_loss, kl_weight * kl_loss





def chunk_wise_recon_loss(target, output, step_size=256):
    """
    Calculate MSE loss with chunk-wise normalization.

    Parameters:
    - target: The target tensor.
    - output: The output tensor.
    - step_size: The size of each chunk.

    Returns:
    - loss: The calculated loss.
    """
    criterion = nn.MSELoss()
    loss = torch.tensor(0.0, device=output.device).float()

    # Flatten the tensors if they have more than 2 dimensions
    if len(output.shape) > 2:
        output = torch.flatten(output, start_dim=1)
        target = torch.flatten(target, start_dim=1)

    n = output.shape[0]
    m = n / step_size
    for i in range(0, n, step_size):
        start_idx = i
        end_idx = min(start_idx + step_size, n)
        tar_tmp = target[:, start_idx:end_idx]
        out_tmp = output[:, start_idx:end_idx]
        loss_tmp = criterion(tar_tmp, out_tmp)
        loss_tmp /= m
        loss += loss_tmp

    return loss
# ─────────────────────────────────────────────────────────────
# Training / Evaluation Loops
# ─────────────────────────────────────────────────────────────

def train(model, optimizer, scaler, train_loader, sampler, rank, n_epochs, save_path):
    if is_main_process(rank):
        os.makedirs(save_path, exist_ok=True)
    # loss_fn = model.module.loss
    global_step = 0
    best_epoch = 0

    best_loss = math.inf
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)  # shuffles shards differently each epoch
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"[Rank {rank}] Epoch {epoch+1}") if is_main_process(rank) else enumerate(train_loader)

        for step, batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):

                batch = batch.to(device)
                # print(batch.shape)
                batch = batch.view(batch.size(0), 3, 1, 64, 64)

                xrec, posterior = model(batch)

            rec_loss= (F.mse_loss(xrec.float(), batch.float(), reduction="mean")+
                    F.l1_loss(xrec.float(), batch.float(), reduction="mean"))

            kl_loss = posterior.kl().mean()

            loss = rec_loss*1000 +  kl_loss * 0.01

            log_dict = {}
            # loss, rec_loss, kl_loss, = vae_loss(xrec.float(), batch.float(), mu.float(), logvar.float(), kl_weight=0.01)

            # x = batch.reshape(-1, 3, 64, 64).float()
            # y = xrec.reshape(-1, 3, 64, 64).float()
            # bloss = F.mse_loss(feature_extractor(x), feature_extractor(y), reduction="sum")
            # loss += bloss

            # self.gl_step += 1
            log_dict["loss"] = loss.item()
            log_dict["rec_loss"] = rec_loss.item()*1000
            log_dict["kl_loss"] = kl_loss.item()  if kl_loss is not None else 0
            # log_dict["bloss"] = bloss.item() if bloss is not None else 0




            global_step += 1

                # loss, logs = model.training_step(batch, step)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

            if is_main_process(rank):
                pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

                # per‑step logging (rank‑0 only) ---------------------------------
            if is_main_process(rank) and wandb and wandb.run is not None:

                wandb.log(log_dict)

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
                torch.save(model.module.state_dict(), os.path.join(save_path, "best_modelvitok_v2.pt"))
                # torch.save(model, os.path.join(save_path, "model_vqvae_best_model_128_ln.pth"))
            print(f"best-epoch-loss--Epoch {best_epoch} | loss={best_loss:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train_epoch_loss": reduced_loss,
            })
        if is_main_process(rank):
            if (epoch + 1) % 10 == 0:
                    print(f'Input: {batch.reshape(batch.shape[0], -1)[0, :10].detach().cpu()},'
                          f' Dec: {xrec.reshape(xrec.shape[0],-1)[0, :10].detach().cpu()}')
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
from vitok.modules.ae import *

if __name__ == "__main__":
    args, unknown = get_parser().parse_known_args()

    # Distributed init ----------------------------------------------------------------
    rank, world_size, local_rank = init_distributed_mode()

    # Reproducibility ------------------------------------------------------------------
    seed_everything(args.seed + rank)  # different seed per rank
    # resnet = models.resnet18(weights="IMAGENET1K_V1")
    # # return the activation **after** layer3’s last ReLU
    # return_nodes = {"layer3.1.relu": "feat"}
    # feature_extractor_base = create_feature_extractor(resnet, return_nodes).cuda()
    #
    # feature_extractor = ResNet18_256(feature_extractor_base).eval()
    # feature_extractor.eval()

    # Config / model -------------------------------------------------------------------
    configs = [OmegaConf.load(args.base)]
    cli_cfg = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(*configs, cli_cfg)
    # model = instantiate_from_config(cfg.model)
    model = AE()


    params = count_trainable_parameters(model)
    base_lr = 1.0e-4

    # resnet = models.resnet18(weights="IMAGENET1K_V1")
    # # return the activation **after** layer3’s last ReLU
    # return_nodes = {"layer3.1.relu": "feat"}
    # feature_extractor_base = create_feature_extractor(resnet, return_nodes).cuda()
    #
    # feature_extractor = ResNet18_256(feature_extractor_base).eval()
    # for p in feature_extractor.parameters():
    #     p.requires_grad = False
    #
    #
    # feature_extractor.eval()

    model.cuda()

    # Wrap in DDP (find_unused_parameters=False unless you need it)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # Optimiser & AMP ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5,
                                  weight_decay=4.0e-5, betas=(0.9, 0.999))
    # optimizer =
    scheduler = WarmUpCosineScheduler(
        optimizer,
        warm_up_steps=0,
        lr_min=0,
        lr_max=1e-4,
        lr_start=base_lr,
        max_decay_steps=10001,
        verbosity_interval=0,
    )
    # scaler = torch.GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    # Dataset / Loader -----------------------------------------------------------------
    # Dataset / Loader -----------------------------------------------------------------
    # train_set = ZooDataset(
    #     root=args.data,
    #     dataset="DeepSeek-R1-Distill-Qwen-1.5B",
    #     split=args.split,
    #     scale=0.1,
    #     tgt=['mlp'],
    #     exd=["norm", ".bias"],
    #     length=196608,
    #     to_image=False,
    #     n_tok=1,
    #     input_size=256,
    #     in_ch=3
    # )
    # train_set= ZooDataset(zoo_root='../zoodatasets_train_1b_subset_256',
    #                       zoo_split="train_1b_subset",
    #                       length=196608,
    #                       resolution=256,
    #                       to_image=True,
    #                       in_channel=3,
    #                       topk=None,
    #                       transform=None,
    #                       scale=1.0)
    # train_set= ZooDataset(zoo_root='../zoodatasets_train_tiny',
    #                       zoo_split="train_tiny",
    #                       length=110592,
    #                       resolution=192,
    #                       to_image=False,
    #                       in_channel=3,
    #                       topk=None,
    #                       transform=None,
    #                       scale=1.0)

    train_set = ZooDataset(zoo_root='../zoodatasets_train_tiny_subset_64_v2',
                           zoo_split="train_tiny_subset-2",
                           length=12288,
                           resolution=64,
                           to_image=True,
                           in_channel=3,
                           topk=None,
                           transform=None,
                           scale=1.0)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        sampler=sampler,
        collate_fn=m_collate,
    )

    # WandB (only initialise on rank‑0) -----------------------------------------------
    if args.wandb and is_main_process(rank) and wandb is not None:
        wandb.init(
            project="vit_small_smoll_vqvae_project",
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
    print(f'=========model===with==:{params}==========parameters')
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

#torchrun --standalone --nnodes=1 --nproc_per_node=4 vqvaetrainer.py \
        # --data modelzoos --dataset joint --split train --n_epochs 1000
        #torchrun --standalone --nnodes=1 --nproc_per_node=1 vqvaetrainer.py
        #torchrun --standalone --nnodes=1 --nproc_per_node=8 vqvaetrainer.py