import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import math
from utils.util import instantiate_from_config
import random

from vqvae_igpg.modules.condmodules import Encoder, Decoder
from vqvae_igpg.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from vqvae_igpg.modules.vqvae.quantize import GumbelQuantize
from vqvae_igpg.modules.vqvae.quantize import EMAVectorQuantizer

class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 cond_stage_config,
                 n_embed,
                 embed_dim,
                 learning_rate,
                 cond_stage_trainable=True,
                 ckpt_path=None,
                 ignore_keys=[],
                 weight_key="weight",
                 cond_key="dataset",
                 anisotropic=False,
                 p_prior=0.5,
                 p_prior_s=0.25,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 device='cuda',
                 *args, **kwargs):
        super().__init__()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_key = weight_key
        self.cond_key = cond_key
        self.anisotropic = anisotropic
        self.p_prior = p_prior
        self.p_prior_s = p_prior_s


        self.cond_stage_trainable = cond_stage_trainable
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)

        # self.input_ch = self.encoder.fch
        # self.input_dim = self.encoder.in_dim
        self.loss.n_classes = n_embed
        self.vocab_size = n_embed
        self.instantiate_cond_stage(cond_stage_config)


        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.initialize_weights()
        # self.init_xavier_uniform()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if monitor is not None:
            self.monitor = monitor

    # self.initialize_weights()
    #
    def initialize_weights(self):
        # Initialize transformer layers and convolutional layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv2d):
                SILU_GAIN = 1.78  # Monte-Carlo estimate for φ=SiLU, Var(x)=1
                fan_in = nn.init._calculate_correct_fan(module.weight, "fan_out")
                bound = math.sqrt(3.0) * SILU_GAIN / math.sqrt(fan_in)
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)



    # def kaiming_silu_conv_(module: nn.Module, mode: str = "fan_out"):
    #     if isinstance(module, nn.Conv2d):
    #         fan_in = nn.init._calculate_correct_fan(module.weight, "fan_out")
    #         bound = math.sqrt(3.0) * SILU_GAIN / math.sqrt(fan_in)
    #         with torch.no_grad():
    #             module.weight.uniform_(-bound, bound)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)

    # def init_xavier_uniform(self):
    #     """
    #     Applies Xavier Uniform initialization to Conv and Linear layers.
    #     """
    #
    #     def _basic_init(module):
    #
    #         if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
    #             nn.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 nn.init.zeros_(module.bias)
    #     self.apply(_basic_init)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=False)
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

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = False
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model


    def interpolate(self, z, scale):
        return torch.nn.functional.interpolate(z, scale_factor=scale, mode='bicubic', align_corners=False)

    def rotate(self, z, angle):
        return torch.rot90(z, k=angle, dims=[-1, -2])

    def process_latent(self, z):
        if self.anisotropic:
            scale_x = random.choice([s / 32 for s in range(8, 32)])
            scale_y = random.choice([s / 32 for s in range(8, 32)])
            scale = (scale_x, scale_y)
        else:
            scale = random.choice([s / 32 for s in range(8, 32)])

        angle = random.choice([1, 2, 3])

        if scale != 1:
            z = self.interpolate(z, scale)
        if angle != 0:
            z = self.rotate(z, angle)


        return z, scale, angle

    def process_image(self, inputs):
        if random.random() < self.p_prior_s:
            scale = random.choice([s / 32 for s in range(8, 32)])
            inputs = self.interpolate(inputs, scale)
        return inputs



    def get_learned_conditioning(self, arch_emb: torch.Tensor, layer_info: list[str], chunk_idx: torch.Tensor, ):
        c = self.cond_stage_model(arch_emb=arch_emb,
                                  layer_info=layer_info,
                                  chunk_idx=chunk_idx,
                                  )
        return c

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    # def latent_to_code(self, h):
    #     quant, emb_loss, info = self.quantize(h)
    #     return quant, emb_loss, info

    def decode(self, quant, c):

        quant = self.post_quant_conv(quant)
        # print(quant.shape, c.shape)
        # print('==========================')
        quant = torch.cat([quant, c], dim=1)
        dec = self.decoder(quant)
        return dec



    def decode_code(self, code_b ):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, inputs, cond, return_indices=False):
        c = self.get_learned_conditioning(**cond)
        quant, diff,  (_,_,ind)= self.encode(inputs)
        dec = self.decode(quant, c)
        if return_indices:
            return  inputs, dec, diff,  ind
        else:
            return inputs,  dec, diff


    def get_input(self, batch, k):
        x = batch[k]
        # x=x.reshape(-1,self.input_ch, self.input_dim)

        return x.float()
    def configure_optimizers(self):
        params = (list(self.encoder.parameters())
                  + list(self.decoder.parameters()) +
                  list(self.quantize.parameters()) +
                  list(self.quant_conv.parameters()) +
                  list(self.post_quant_conv.parameters()))

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        optimizer = torch.optim.AdamW(params,
                                     lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=4e-2)
        # optimizer = torch.optim.SGD(params,
        #                               lr=self.learning_rate, momentum=0.9,)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-7, last_epoch=-1)
        return  optimizer

    def get_last_layer(self):
        return self.decoder.conv_out.weight


    def sample_x(self, x, temp=0.1):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, temp=temp)
        dec = self.decode(quant)
        return dec, info



class GumbelVQ(VQModel):
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
        self.reg = 0.0
        self.global_step = 0

        self.quantize = GumbelQuantize(embed_dim, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)  # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self, global_step):
        self.quantize.temperature = self.temperature_scheduler(global_step)
        return self.quantize.temperature

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError


    def configure_optimizers(self):
        params = (list(self.encoder.parameters())
                  + list(self.decoder.parameters()) +
                  list(self.quantize.parameters()) +
                  list(self.quant_conv.parameters()) +
                  list(self.post_quant_conv.parameters()))

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        optimizer = torch.optim.AdamW(params,
                                     lr=self.learning_rate, betas=(0.5, 0.9))
        # optimizer = torch.optim.SGD(params,
        #                               lr=self.learning_rate, momentum=0.9,)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-7, last_epoch=-1)
        return  optimizer

class GumbelVQNoDisc(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 cond_stage_config,
                 n_embed,
                 embed_dim,
                 learning_rate,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 weight_key="weight",
                 monitor=None,
                 kl_weight=0.1,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig=ddconfig,
                         lossconfig=lossconfig,
                         learning_rate=learning_rate,
                         cond_stage_config=cond_stage_config,
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
        self.cond_stage_trainable=True


        self.quantize = GumbelQuantize(embed_dim, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)  # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self, global_step):
        self.quantize.temperature = self.temperature_scheduler(global_step)
        return self.quantize.temperature

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

        xrec, qloss, indices = self(x, return_indices=True)
        # print(xrec.shape, qloss.shape)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train",predicted_indices=indices)
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
        xrec, qloss, indices = self(x, return_indices=True)
        # xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val",predicted_indices=indices)

        rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #          prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/aeloss", aeloss,
        #          prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log_dict(log_dict_ae)
        return rec_loss, log_dict_ae

    def configure_optimizers(self):
        params = (list(self.encoder.parameters())
                  + list(self.decoder.parameters()) +
                  list(self.quantize.parameters()) +
                  list(self.quant_conv.parameters()) +
                  list(self.post_quant_conv.parameters()))

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        optimizer = torch.optim.AdamW(params,
                                     lr=self.learning_rate, betas=(0.5, 0.9))
        # optimizer = torch.optim.SGD(params,
        #                               lr=self.learning_rate, momentum=0.9,)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-7, last_epoch=-1)
        return  optimizer

class EMAVQ(VQModel):
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
                 ):
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
        self.reg = 0.0
        self.global_step = 0
        self.learning_rate = learning_rate

        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           )


    def training_step(self, batch, batch_idx):
        # self.temperature_scheduling()
        x = self.get_input(batch, self.weight_key).to(self.device)
        # self.global_step += 1
        # alpha = torch.rand(1)
        # if alpha > 0.5:
        #     x = torch.randn_like(x, device=x.device)

        xrec, qloss = self(x)
        # print(xrec.shape, qloss.shape)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        log_dict_ae['train/temp_t'] =1
        # self.log("train/aeloss", aeloss,
        #          prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        #
        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        # self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True,
        #          on_epoch=True)
        # mse = F.mse_loss(xrec, x) * 1000
        return aeloss, log_dict_ae  # + mse

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
                                     list(self.quant_conv.parameters()) +
                                     list(self.post_quant_conv.parameters()),
                                     lr=self.learning_rate, betas=(0.5, 0.9))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-7, last_epoch=-1)
        return optimizer

        # def configure_optimizers(self):
        #     lr = self.learning_rate
        #     # Remove self.quantize from parameter list since it is updated via EMA
        #     opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
        #                               list(self.decoder.parameters()) +
        #                               list(self.quant_conv.parameters()) +
        #                               list(self.post_quant_conv.parameters()),
        #                               lr=lr, betas=(0.5, 0.9))
        #     opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                                 lr=lr, betas=(0.5, 0.9))
        #     return [opt_ae, opt_disc], []


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