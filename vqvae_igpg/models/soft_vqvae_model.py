import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn

from utils.util import instantiate_from_config

from vqvae_igpg.modules.modules import Encoder, Decoder
from vqvae_igpg.modules.vqvae.svqvae import SoftVectorQuantizer
from vqvae_igpg.modules.vqvae.quantize import EMAVectorQuantizer, VectorQuantizer

class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 learning_rate,
                 ckpt_path=None,
                 ignore_keys=[],
                 weight_key="weight",
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 device='cuda',
                 *args, **kwargs):
        super().__init__()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_key = weight_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.input_ch = self.encoder.fch
        self.input_dim = self.encoder.in_dim



        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        # sd = torch.load(path, map_location="cpu")["state_dict"]
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
        quant, diff, _ = self.encode(input)

        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        # x=x.reshape(-1,self.input_ch, self.input_dim)

        return x.float()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                     list(self.decoder.parameters()) +
                                     list(self.quantize.parameters()) +
                                     list(self.quant_conv.parameters()) +
                                     list(self.post_quant_conv.parameters()),
                                     # list(self.encode_layer.parameters()),
                                     lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer

    def get_last_layer(self):
        return self.decoder.conv_out.weight


    def sample_x(self, x, temp=0.1):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, temp=temp)
        dec = self.decode(quant)
        return dec, info



class SoftVVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 softconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 weight_key="weight",
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         weight_key=weight_key,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = instantiate_from_config(softconfig)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

class SoftVQNoDisc(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 learning_rate,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 weight_key="weight",
                 monitor=None,
                 kl_weight=1e-3,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig=ddconfig,
                         lossconfig=lossconfig,
                         learning_rate=learning_rate,
                         n_embed=n_embed,
                         embed_dim=embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         weight_key=weight_key,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed
        self.reg =0.0
        self.global_step=0

        self.quantize = instantiate_from_config(softconfig)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)  # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.weight_key).to(self.device)
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


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.weight_key)
        xrec, qloss = self(x)
        # xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")

        rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #          prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/aeloss", aeloss,
        #          prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log_dict(log_dict_ae)
        return rec_loss, log_dict_ae

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                     list(self.decoder.parameters()) +
                                     list(self.quantize.parameters()) +
                                     list(self.quant_conv.parameters()) +
                                     list(self.post_quant_conv.parameters()),
                                     lr=self.learning_rate, betas=(0.5, 0.9))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-7, last_epoch=-1)
        return  optimizer
#
# class EMAVQ(VQModel):
#     def __init__(self,
#                  ddconfig,
#                  lossconfig,
#                  n_embed,
#                  embed_dim,
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  image_key="image",
#                  colorize_nlabels=None,
#                  monitor=None,
#                  remap=None,
#                  sane_index_shape=False,  # tell vector quantizer to return indices as bhw
#                  ):
#         super().__init__(ddconfig,
#                          lossconfig,
#                          n_embed,
#                          embed_dim,
#                          ckpt_path=None,
#                          ignore_keys=ignore_keys,
#                          image_key=image_key,
#                          colorize_nlabels=colorize_nlabels,
#                          monitor=monitor,
#                          )
#         self.quantize = EMAVectorQuantizer(n_embed=n_embed,
#                                            embedding_dim=embed_dim,
#                                            beta=0.25,
#                                            remap=remap)
#     def configure_optimizers(self):
#         lr = self.learning_rate
#         #Remove self.quantize from parameter list since it is updated via EMA
#         opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
#                                   list(self.decoder.parameters())+
#                                   list(self.quant_conv.parameters())+
#                                   list(self.post_quant_conv.parameters()),
#                                   lr=lr, betas=(0.5, 0.9))
#         opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
#                                     lr=lr, betas=(0.5, 0.9))
#         return [opt_ae, opt_disc], []