import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import log_cosh_error
from typing import Tuple
from vqvae_igpg.modules.transformer_modules.llama_encoder import Encoder, Decoder, WeightDecode, WeightEmbed, TransformerBlock
from vqvae_igpg.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from vqvae_igpg.modules.vqvae.vitdvq import GumbelQuantize1D as  GumbelQuantize
from vqvae_igpg.modules.vqvae.quantize import EMAVectorQuantizer

from utils.util import instantiate_from_config




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
    #
    # # Compute squared errors only where mask is 1, then average over the total number of valid elements.
    # loss_sum = (((recon - x)**2) * mask).sum()
    # mask_sum = mask.sum() + 1e-8  # to avoid division by zero
    # recon_loss = (loss_sum / mask_sum)*1000.0

    recon_loss = F.mse_loss(recon, x, reduction="mean") * 1000 + mmd_loss(x, recon) * 100
    # recon_loss = log_cosh_error(recon,x).mean()*1000.0
    # recon_loss = F.l1_loss(recon, x, reduction="mean") * 1000.0

    # KL divergence loss (averaged over the batch)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss * kl_weight


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



class GumbelVQNoDisc(nn.Module):
    def __init__(self,
                 learning_rate,
                 enconfig,
                 deconfig,
                 temperature_scheduler_config,
                 lossconfig,
                 embed_dim,
                 n_embed,
                 codebook_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="weight",
                 cond_key="dataset",
                 freeze_encoder=False,
                 freeze_decoder=False,
                 kl_weight=1.0,
                 device='cuda',
                 use_ema=False,
                 enc_type='vit',
                 monitor=None,
                 remap=None,
                 channel_last=False,
                 sane_index_shape=False,
                 ):
        super().__init__()
        self.devices = device
        self.cond_key = cond_key
        self.learning_rate = learning_rate
        self.num_tokens = math.ceil(enconfig.length / enconfig.chunk_size)
        self.input_key = input_key
        self.length = enconfig.length
        self.use_ema = use_ema
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.channel_last = channel_last
        self.n_embed = n_embed
        self.global_step =0
        self.loss = instantiate_from_config(lossconfig)
        self.enc_type = enc_type
        self.codebook_dim = codebook_dim
        self.encoder = Encoder(**enconfig).to(device)
        self.decoder = Decoder(**deconfig).to(device)
        self.quantize = GumbelQuantize(codebook_dim, codebook_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       channels_last=channel_last,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)

        if enc_type == 'cnn':
            raise NotImplementedError
            # self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
            # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        else:
            if self.channel_last:
                self.quant_conv = nn.Linear(embed_dim, codebook_dim)
                self.post_quant_conv = nn.Linear(codebook_dim, embed_dim)

            else:
                self.quant_conv = torch.nn.Conv1d(self.num_tokens, codebook_dim, 1)
                self.post_quant_conv = torch.nn.Conv1d(codebook_dim, self.num_tokens, 1)

        # self.quant_conv = nn.Linear(embed_dim, codebook_dim)
        # self.post_quant_conv = nn.Linear(codebook_dim, embed_dim)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        # for p in self.encoder.parameters():
        #     p.requires_grad=False

        if freeze_encoder:
            self.freeze(self.encoder)
        if freeze_decoder:
            self.freeze(self.decoder)


    def freeze(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = False
            model.eval()

    def xavier_initialize(self, model):
        """
        Applies Xavier initialization to all layers in the given model.
        Args:
            model (nn.Module): The neural network model to initialize.
        """
        for name, param in model.named_parameters():
            if "weight" in name:
                if param.dim() > 1:  # Only apply to layers with at least 2 dimensions
                    nn.init.xavier_uniform_(param)
                    print(f"Xavier initialized: {name}")
            elif "bias" in name:
                nn.init.zeros_(param)
                print(f"Bias initialized to zero: {name}")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        if isinstance(input, dict):
            input = input[self.input_key].to(self.device)
        quant, diff, _ = self.encode(input)
        # print(quant.shape)
        dec = self.decode(quant)
        # aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        return x

    def training_step(self, batch, batch_idx):
        self.temperature_scheduling(self.global_step)
        x = self.get_input(batch, self.input_key).to(self.device)
        self.global_step += 1
        # alpha = torch.rand(1)
        # if alpha > 0.5:
        #     x = torch.randn_like(x, device=x.device)

        xrec, qloss = self(x)
        # print(xrec.shape, qloss.shape)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        log_dict_ae['train/temp_t']= self.quantize.temperature
        # self.log("train/aeloss", aeloss,
        #          prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        #
        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        # self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True,
        #          on_epoch=True)
        # mse = F.mse_loss(xrec, x) * 1000
        return aeloss, log_dict_ae #+ mse

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) +
                                      list(self.decoder.parameters()) +
                                      list(self.quantize.parameters()) +
                                      list(self.quant_conv.parameters()) +
                                      list(self.post_quant_conv.parameters()),
                                      lr=self.learning_rate, betas=(0.5, 0.9))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-7, last_epoch=-1)
        return optimizer

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def temperature_scheduling(self, global_step):
        self.quantize.temperature = self.temperature_scheduler(global_step)
        return self.quantize.temperature

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError


    def sample_x(self, x, temp=0.1):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, temp=temp)
        dec = self.decode(quant)
        return dec, info




class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

