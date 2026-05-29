import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from torch.linalg import multi_dot
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset
from functools import partial
from PIL import Image
from helpers.helpers import *
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
# from zoodatasets.zoo_datasets import ZooDataset
from zoodatasets.tf_datasets import ZooDataset
from helpers.misc import progress_bar
from utils.util import instantiate_from_config
import yaml

# Import wandb and set to offline mode
import wandb
# os.environ["WANDB_MODE"] = "offline"

import random  # for demo script
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms as transforms

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('--data', default='modelzoos', type=str, help='dataset root')
    parser.add_argument('--data_root', default='../Datasets/', type=str, help='dataset root for cifar10, mnist, ..')
    parser.add_argument('--topk', default=30, type=int, help='number of sample per dataset in training loader')
    parser.add_argument('--dataset', default='joint', type=str, help='dataset choice amoung'
                                                                     ' [mnist, svhn, cifar10, stl10, joint')
    parser.add_argument('--split', default='train', type=str, help='dataset split{ train, test, val]')
    parser.add_argument('--ae_type', default='ldm', type=str, help='auto encoder type [ldm, vqvae, simple]')
    parser.add_argument('--save_path', default='autocheckpoints', type=str, help='checkpointys folders')
    parser.add_argument('--gpus', default=0, type=int, help='device')
    # parser.add_argument('--num_workers', default=4, type=int, help='device')

    parser.add_argument('--n_epochs', default=1000000, type=int, help='max epoch')
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="adt",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.simple_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",

        # default="vqvae_igpg/configs/first_stage_config_vqvae.yaml",
        # default="stage1/configs/fisrt_stage_soft_vqvae_.simple_config.yaml",
        # default="stage1/configs/full_small_llama_config.simple_config.yaml",
        # default="vqvae_igpg/configs/first_stage_ema_config_kl.yaml",
        default="vqvae_igpg/configs/llm_vitvqvae_config.yaml",
        # default="stage1/configs/ default="stage1/configs/fisrt_stage_soft_vqvae_.simple_config.yaml",",

    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser
#
def seed_everything(seed=1234):
    import random, os
    import numpy as np
    import torch

    random.seed(0)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def my_loss(output, target):
    ed =0
    n = 256*256*3
    step = 8192
    loss = 0.0
    for i in range(n, step):
        ed = i+step
        loss+=F.mse_loss(output[:, i:ed] - target[: i:ed])/(torch.std(output[:, i:ed]))
    # loss = torch.mean((output[:, i:ed] - target[: i:ed])**2)
    return loss
# my_loss  = my_loss()
# ▶ helpers/grad_monitor.py
from collections import defaultdict
import torch, math

class GradMonitor:
    def __init__(self, model, every=100, writer=None, prefix="grad"):
        self.model  = model
        self.every  = every          # log frequency (steps)
        self.writer = writer         # e.g. wandb / tb.SummaryWriter
        self.step   = 0

        self.handles = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            h = p.register_hook(self._make_hook(name))
            self.handles.append(h)

        self.bucket = defaultdict(list)

    def _make_hook(self, name):
        def hook(grad):
            if grad is None:
                return
            self.bucket[name].append(grad.detach())
        return hook

    @torch.no_grad()
    def flush(self):
        if self.step % self.every:
            return

        for name, grads in self.bucket.items():
            g = torch.cat([g.reshape(-1) for g in grads])
            l2 = g.norm()                             # ‖∇θ‖₂
            p  = dict(self.model.named_parameters())[name]
            gwr = l2 / (p.data.norm() + 1e-12)        # GWR

            if self.writer:
                self.writer.log({f"{name}/grad_l2": l2.item(),
                                  f"{name}/grad_to_weight": gwr.item()},
                                 step=self.step)

        self.bucket.clear()

    def step_end(self):
        self.step += 1

    def close(self):
        for h in self.handles: h.remove()



def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def train(model, optimizer, n_epochs, traindataloader, testdataloader=None):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    bloss = 100
    btest = 2.0
    use_amp=True
    cr =[]
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 5)

    bloss = 100.0
    use_amp = True
    btest = 2.0
    # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    # monitor = GradMonitor(model, every=50, writer=wandb)

    for epoch in range(n_epochs):
        print(f'\nEpoch: {epoch + 1}/{n_epochs}')
        model.train()
        train_loss = 0
        total = 0

        # Initialize tqdm progress bar for the training loop
        progress_bar = tqdm(enumerate(traindataloader), total=len(traindataloader), desc=f"Epoch {epoch + 1}")

        for batch_idx, inputs in progress_bar:
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=use_amp):
                loss, logs = model.training_step(inputs, batch_idx)

            # Backward pass and optimization step
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # monitor.flush()  # ← logs every 50 steps
            # scaler.step(optimizer)
            # scaler.update()
            # monitor.step_end()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            # torch.nn.utils.clip_grad_value_(model.parameters(), 1)  # 3️⃣ value clip
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # 2️⃣ norm clip
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            total += inputs['weight'].size(0)

            # Update tqdm progress bar
            progress_bar.set_postfix({
                'Loss': f"{train_loss / (batch_idx + 1):.4f}",
                'LR': f"{optimizer.param_groups[-1]['lr']:.6f}"
            })
            # Print additional loss details
        rec_loss = logs['train/rec_loss']
        kld_loss = logs['train/quant_loss']
        nnl_loss = logs['train/total_loss']
        temp_t = logs['train/temp_t']
        # log_var = logs['train/logvar']


        tloss = train_loss / len(traindataloader)

        # Save model with the best training loss
        if bloss > tloss:
            bloss = tloss
            print(f'Saving model with best training loss: {bloss:.4f}')
            torch.save(model, os.path.join(args.save_path, f'vq_model_test_vqvae_subsetdeepseek_ema_k.pth'))


        print(f'Best Training Loss: {bloss:.4f}, LR: {optimizer.param_groups[-1]["lr"]:.6f}')
        print(f'Rec Loss: {rec_loss}, quant Loss: {kld_loss}, total Loss: {nnl_loss}')

        # Perform model evaluation every 100 epochs
        if (epoch + 1) % 10 == 0:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                model.eval()
                inputr = inputs['weight']
                dec, _ = model(inputs['weight'].to(device))
                dec = dec.reshape(dec.size(0), -1)
                # dec = dec.argmax(dim=-1) - 1

                print(f'Input: {inputr[0,:10].detach().cpu()}, Dec: {dec[0,:10].detach().cpu()}')
                # recon_error = torch.nn.functional.mse_loss(dec, inputr)
                # print(f'Recon Error: {recon_error}')
        wandb.log({
            "epoch": epoch + 1,
            "train_epoch_loss": tloss,
            "rec_loss": rec_loss.item(),
            "kl_loss": kld_loss.item(),
            "nll_loss": nnl_loss.item(),
            "temp_t":temp_t,
        })

    #
    #

        if bloss > tloss:
            bloss = tloss
            print(f'saving best training loss is:{bloss}')
            torch.save(model, os.path.join(args.save_path,f'chunk_vit_zoo.pth'))
        print(f'best training loss is:{bloss}')
        #best_top_old_leader_.pth to be tested



def evaluate(model , testdataloader):
    model.eval()
    test_loss = 0
    idx = 1
    with torch.no_grad():
        for batch_idx, inputs in enumerate(testdataloader):
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            loss = model.validation_step(inputs, batch_idx)
            # inputs = inputs.to(device)
            outputs, _ = model(inputs)
            recon_error = F.mse_loss(outputs, inputs) * 1000
            loss = recon_error
            test_loss += loss.item()
            progress_bar(batch_idx, len(testdataloader), 'Loss: %.6f |'
                         % (test_loss / (batch_idx + 1)))
            idx = batch_idx+1
        tloss =(test_loss / idx)
    return  tloss


def add_to_config(mydict, cfl="./Experiments/stage1/configs/base_config_imnet_kl.simple_config.yaml"):
    with open(cfl, 'w') as configfile:
        data = yaml.dump(mydict, configfile, indent=4, sort_keys=False)
        print("Write successful")

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def m_collate(batch):
    sample = {}

    data = [item['weight'] for item in batch]

    data = torch.cat(data, 0).type(torch.float32)

    return data
def lr_lambda(current_step: int, warmup_iters=50):
    if current_step < warmup_iters:
        return current_step / max(1, warmup_iters)
    return 1.0
def count_trainable_parameters(model: nn.Module) -> int:
    """Computes the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(file_path):
    import yaml
    with open(file_path, "r") as f:
        return yaml.safe_load(f)



def m_collate(batch):
    sample = {}
    data = [item['weight'] for item in batch]
    data = torch.cat(data, 0)
    return data

# from stage1.modules.losses.CustomLosses import LayerWiseReconLoss, ChunkWiseReconLoss


if __name__ == "__main__":
    # seed_everything(seed=1234)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # sys.path.append(os.getcwd())
    parser = get_parser()
    use_amp = True

    args = parser.parse_args()
    opt, unknown = parser.parse_known_args()


    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainset = ZooDataset(root=args.data, dataset="Llama-3.2-1B-Instruct", split=args.split,
                          scale=0.025, tgt=['k_proj'], exd=['norm', '.bias'], length=49152, n_tok=1)
    traindataloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=8)
    # testdataloader = DataLoader(valset, shuffle=False, batch_size=4, num_workers=4)

    # parser = Trainer.add_argparse_args(parser)

    nowname= opt.name+now
    # seed_everything(opt.seed)
    print(opt.base)
    print('----------------------')
    configs = [OmegaConf.load(opt.base)]
    myconfig= load_config(opt.base)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    model = instantiate_from_config(config.model)
    model = model.to(device)
    model.device = device
    optimizer = model.configure_optimizers()
    #
    # initial_lr = model.learning_rate
    # # Number of warmup iterations
    # warmup_iters = 50
    # # Number of total iterations (epochs * iterations per epoch)
    # total_iters = 100000
    # # Linear warmup scheduler
    # scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # # Cosine annealing scheduler after warmup
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_iters - warmup_iters))
    # # Combine schedulers using SequentialLR
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
    #                          milestones=[warmup_iters])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-7, last_epoch=-1)
    criterion = model.loss
    # train(model, optimizer, args.n_epochs, traindataloader, testdataloader)
    print("Trainable parameters:", count_trainable_parameters(model))
    wandb.init(
        project="_sweep_128_vqvae_trainer-project",
        name=f"{args.name}_{now}",
        config={
            "epochs": args.n_epochs,
            "batch_size": 64,
            "learning_rate": optimizer.param_groups[0]['lr'],
        },
        mode="online",  # crucial for offline usage
    )
    train(model, optimizer, args.n_epochs, traindataloader)

