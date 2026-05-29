import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import wandb
from pytorch_lightning.loggers import WandbLogger

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4, 5, 6, 7"


def to_ternary(x):
    # x: tensor (any shape), float
    x = x.clone()
    x[x < -0.5] = -1
    x[(x >= -0.5) & (x < 0.5)] = 0
    x[x >= 0.5] = 1
    return x.type(torch.int)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


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
    parser.add_argument('--data', default='../Datasets', type=str, help='dataset root')
    parser.add_argument('--data_root', default='../Datasets/', type=str, help='dataset root for cifar10, mnist, ..')
    parser.add_argument('--topk', default=30, type=int, help='number of sample per dataset in training loader')
    parser.add_argument('--dataset', default='joint', type=str, help='dataset choice among [mnist, svhn, cifar10, stl10, joint]')
    parser.add_argument('--split', default='train', type=str, help='dataset split {train, test, val}')
    parser.add_argument('--ae_type', default='ldm', type=str, help='auto encoder type [ldm, vqvae, simple]')
    parser.add_argument('--save_path', default='vae_checkpoints', type=str, help='checkpoints folder')
    parser.add_argument('--gpus', default=0, type=int, help='device')
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument('--n_epochs', default=10000, type=int, help='max epoch')
    parser.add_argument('--zoo_root', default='modelzoos', help='zoo file folder')
    parser.add_argument('--zoo_name', default="train_7_8B_models",  help='zoo split name')
    parser.add_argument('--exd', default=['.bias', 'norm'], help='skip layer')
    parser.add_argument('--tgt', default=None,  help='targets')
    parser.add_argument('--length', default=196608, type=int, help='chunk size')
    parser.add_argument('--zoo_file', default='zoo_config.simple_config.yaml', type=str, help='zoo file')
    parser.add_argument('--sizes', default=[3, 256], help='shape to image')
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
        help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.",
        # default="vqvae_igpg/configs/linear_base_vqvae_config_kl.simple_config.yaml", #good
        default="vqvae_igpg/configs/base_vqvae_model_config.yaml",

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
        help="directory for logging",
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

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))






if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    sys.path.append(os.getcwd())

    parser = get_parser()
    # parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    nowname= opt.name+now
    #
    # seed_everything(opt.seed)
    print(opt.base)
    print('----------------------')
    configs = [OmegaConf.load(opt.base)]
    # init and save configs
    # configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)


    model = instantiate_from_config(config.model)

#917504 llama ffn

    ds = instantiate_from_config(config.data)
    # ds.prepare_data()
    ds.setup(stage='fit')

    print("#### Data #####")
    # print(f'dataset {ds.dataset}')
    # trainer = pl.Trainer( accumulate_grad_batches=4, accelerator="gpu", devices=1, min_epochs=10000,
    #                       max_epochs=100000)

    wandb_logger = WandbLogger(
        name=nowname,
        project=opt.project if opt.project else "ldm_vqvae-training",
        save_dir=opt.logdir,
        offline=False,  # Remove or set to False if you want online logging
        entity=None,  # Fill in if using a team account
        log_model=False,  # Set True to log checkpoints
    )
    checkpoint_callback = ModelCheckpoint(monitor='train/aeloss',
                                          dirpath='vgvae_checkpoints/',
                                          filename='checkpoint_8b_models_block_{epoch}_',
                                          every_n_epochs=1
                                          )

    trainer = pl.Trainer(accelerator="gpu", devices=-1, min_epochs=10000, precision="bf16",
                         strategy="ddp",
                         max_epochs=1000000, log_every_n_steps=1, callbacks=[checkpoint_callback],
                         logger=wandb_logger,
                         )
    trainer.fit(model, ds)