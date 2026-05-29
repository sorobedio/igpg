import torch
import torch.nn as nn

import torch.nn.functional as F
from numpy.ma.core import masked


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def matpadder(x, max_in=512):
    shape =x.shape
    # delta1 = max_in - shape[0]
    delta2 = max_in - shape[1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out

def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


class Myloss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, n_classes=16):

        super().__init__()
        self.kl_weight = kl_weight
        self.n_classes = n_classes
        # self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        # self.ploss = PerceptualLoss()

    def forward(self, inputs, reconstructions, posteriors, split="train",weights=None,  predicted_indices=None):
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        reconstructions = reconstructions.contiguous().view_as(inputs)
        rec_loss = (inputs.contiguous() - reconstructions.contiguous())**2

        nll_loss = rec_loss
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = weighted_nll_loss + self.kl_weight * kl_loss #+ self.ploss(reconstructions, inputs)*10


        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               # "{}/logvar".format(split): self.logvar.detach(),
               "{}/kl_loss".format(split): kl_loss.detach().mean(),
               "{}/nll_loss".format(split): nll_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               }
        if predicted_indices is not None:
            assert self.n_classes is not None
            with torch.no_grad():
                perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
            log[f"{split}/perplexity"] = perplexity
            log[f"{split}/cluster_usage"] = cluster_usage
        return loss, log


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).cuda()
        # self.feature_extractor = feature_extractor
        self.max_len = 3072

    def forward(self, input, target):
        input = matpadder(input, max_in=3072).reshape(-1, 3, 32, 32)
        target = matpadder(target, max_in=3072).reshape(-1, 3, 32, 32)
        input_features =  self.feature_extractor(input)
        target_features =  self.feature_extractor(target)
        loss = 0
        for inp_feat, tar_feat in zip(input_features, target_features):
            loss += F.mse_loss(inp_feat, tar_feat)
        return loss



class LayerWiseReconLoss(nn.Module):
    """
    MSE w/ layer-wise normalization
    """

    def __init__(self, config_path, step_size=1024):
        super(LayerWiseReconLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.step_size=step_size
        self.loss_mean = None
        self.layer_info = torch.load(config_path)

    def forward(self, output, target):
        # check validity
        assert (
            output.shape == target.shape
        ), f"Input shape mismatch. output {output.shape} vs target {target.shape}"

        loss = torch.tensor(0.0, device=output.device).float()

        layers = list(self.layer_info)
        for l in layers:
            start_idx = self.layer_info[l]['idx_start']
            end_idx = self.layer_info[l]['idx_end']
            tar_tmp = target[:, start_idx:end_idx]
            out_tmp = output[:, start_idx:end_idx]
            loss_tmp = self.criterion(out_tmp, tar_tmp)
            loss_tmp /= output.shape[0]
            loss += loss_tmp


        return loss


class MyVQLoss(nn.Module):
    def __init__(self, codebook_weight=1.0, loss_type='l2'):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.loss_type=loss_type
        self.huber =  torch.nn.SmoothL1Loss(beta=0.05, reduction="mean")  # Huber loss
        # self.pad_value = 0
        # if sel
        # self.pixel_weight = pixelloss_weight
    def forward(self, codebook_loss, inputs, reconstructions, split="train"):
        reconstructions = reconstructions.contiguous().view_as(inputs)
        # mask = (inputs!=self.pad_value).float()
        if self.loss_type=='l2':
            rec_loss = F.mse_loss(reconstructions, inputs, reduction="sum")
        else:
            rec_loss =F.l1_loss(reconstructions, inputs, reduction="sum")
        rec_loss = rec_loss+ self.huber(reconstructions, inputs)
        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        # nll_loss = torch.sum(nll_loss) / torch.sum(mask)
        loss = nll_loss*1000.0 + self.codebook_weight * codebook_loss.mean()
        # print(loss)
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/quant_loss".format(split): codebook_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean()*1000,
               }
        return loss, log



class VQLoss(nn.Module):
    def __init__(self, codebook_weight=1.0, loss_type='l1', n_classes=10):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.loss_type=loss_type
        # self.pad_value = 0
        self.n_classes = n_classes
        # if sel
        # self.pixel_weight = pixelloss_weight
    def forward(self, codebook_loss, inputs, reconstructions, split="train", predicted_indices=None):
        reconstructions = reconstructions.contiguous().view_as(inputs)
        # mask = (inputs!=self.pad_value).float()
        if self.loss_type=='l2':
            rec_loss = (inputs.contiguous() - reconstructions.contiguous())**2
        else:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        # nll_loss = torch.sum(nll_loss) / torch.sum(mask)
        loss = nll_loss + self.codebook_weight * codebook_loss.mean()
        # print(loss)
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/quant_loss".format(split): self.codebook_weight*codebook_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean()
               }
        if predicted_indices is not None:
            assert self.n_classes is not None
            with torch.no_grad():
                perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
            log[f"{split}/perplexity"] = perplexity
            log[f"{split}/cluster_usage"] = cluster_usage
        return loss, log



#
#
# class VQLPIPSWithDiscriminator(nn.Module):
#     def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
#                  disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
#                  perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
#                  disc_ndf=64, disc_loss="hinge"):
#         super().__init__()
#         assert disc_loss in ["hinge", "vanilla"]
#         self.codebook_weight = codebook_weight
#         self.pixel_weight = pixelloss_weight
#         self.perceptual_loss = LPIPS().eval()
#         self.perceptual_weight = perceptual_weight
#
#         self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
#                                                  n_layers=disc_num_layers,
#                                                  use_actnorm=use_actnorm,
#                                                  ndf=disc_ndf
#                                                  ).apply(weights_init)
#         self.discriminator_iter_start = disc_start
#         if disc_loss == "hinge":
#             self.disc_loss = hinge_d_loss
#         elif disc_loss == "vanilla":
#             self.disc_loss = vanilla_d_loss
#         else:
#             raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
#         print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
#         self.disc_factor = disc_factor
#         self.discriminator_weight = disc_weight
#         self.disc_conditional = disc_conditional
#
#     def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
#         if last_layer is not None:
#             nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
#             g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
#         else:
#             nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
#             g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]
#
#         d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
#         d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
#         d_weight = d_weight * self.discriminator_weight
#         return d_weight
#
#     def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
#                 global_step, last_layer=None, cond=None, split="train"):
#         rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
#         if self.perceptual_weight > 0:
#             p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
#             rec_loss = rec_loss + self.perceptual_weight * p_loss
#         else:
#             p_loss = torch.tensor([0.0])
#
#         nll_loss = rec_loss
#         #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
#         nll_loss = torch.mean(nll_loss)
#
#         # now the GAN part
#         if optimizer_idx == 0:
#             # generator update
#             if cond is None:
#                 assert not self.disc_conditional
#                 logits_fake = self.discriminator(reconstructions.contiguous())
#             else:
#                 assert self.disc_conditional
#                 logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
#             g_loss = -torch.mean(logits_fake)
#
#             try:
#                 d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
#             except RuntimeError:
#                 assert not self.training
#                 d_weight = torch.tensor(0.0)
#
#             disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
#             loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
#
#             log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
#                    "{}/quant_loss".format(split): codebook_loss.detach().mean(),
#                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
#                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
#                    "{}/p_loss".format(split): p_loss.detach().mean(),
#                    "{}/d_weight".format(split): d_weight.detach(),
#                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
#                    "{}/g_loss".format(split): g_loss.detach().mean(),
#                    }
#             return loss, log
#
#         if optimizer_idx == 1:
#             # second pass for discriminator update
#             if cond is None:
#                 logits_real = self.discriminator(inputs.contiguous().detach())
#                 logits_fake = self.discriminator(reconstructions.contiguous().detach())
#             else:
#                 logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
#                 logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
#
#             disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
#             d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
#
#             log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
#                    "{}/logits_real".format(split): logits_real.detach().mean(),
#                    "{}/logits_fake".format(split): logits_fake.detach().mean()
#                    }
#             return d_loss, log
#
#
#
