import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
import numpy as np


class GumbelQuantize1D(nn.Module):
    """
    Gumbel-Softmax quantiser for 1-D sequences.

    Supports *either* input layout:
      • channels-first  (B, E, N)   – set channels_last=False   (default)
      • channels-last   (B, N, E)   – set channels_last=True
    """

    def __init__(
        self,
        num_hiddens: int,          # E (embedding dim)
        embedding_dim: int,        # D (output latent dim)
        n_embed: int,              # K (codebook size)
        *,
        channels_last: bool = False,
        straight_through: bool = True,
        kl_weight: float = 5e-4,
        temp_init: float = 1.0,
        use_vqinterface: bool = True,
        remap=None,
        unknown_index: str | int = "random",
    ):
        super().__init__()

        self.channels_last   = channels_last
        self.embedding_dim   = embedding_dim
        self.n_embed         = n_embed
        self.straight_through= straight_through
        self.temperature     = temp_init
        self.kl_weight       = kl_weight
        self.use_vqinterface = use_vqinterface

        # Projection: Conv1d for channels-first, Linear for channels-last
        if channels_last:
            self.proj = nn.Linear(num_hiddens, n_embed)      # (B,N,E) → (B,N,K)
        else:
            self.proj = nn.Conv1d(num_hiddens, n_embed, 1)   # (B,E,N) → (B,K,N)

        self.embed = nn.Embedding(n_embed, embedding_dim)

        # Optional remapping table (unchanged)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed      = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed     += 1
            print(f"Remapping {n_embed} → {self.re_embed}. "
                  f"Unknown → {self.unknown_index}.")
        else:
            self.re_embed = n_embed

    # --------------------------------------------------------------------- #
    def _remap_to_used(self, inds):
        flat  = inds.view(inds.size(0), -1)
        used  = self.used.to(flat)
        match = (flat[..., None] == used).long()
        new   = match.argmax(-1)
        unknown = match.sum(-1) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(
                0, self.re_embed, new[unknown].shape, device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.view_as(inds)

    def _unmap_to_all(self, inds):
        flat = inds.view(inds.size(0), -1)
        used = self.used.to(flat)
        if self.re_embed > self.used.shape[0]:          # extra token
            flat[flat >= self.used.shape[0]] = 0
        back = torch.gather(used[None].expand(flat.size(0), -1), 1, flat)
        return back.view_as(inds)

    # --------------------------------------------------------------------- #
    def forward(self, z, *, temp=None, return_logits=False):
        """
        z  : (B,E,N)  if channels_last=False   – channels-first
           : (B,N,E)  if channels_last=True    – channels-last
        """
        # print(z.shape)
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        # --- Project to codebook logits ----------------------------------- #
        if self.channels_last:
            logits = self.proj(z)                              # (B,N,K)
            logits = logits.permute(0, 2, 1)                  # → (B,K,N)
        else:
            # z =z .permute(0, 2, 1)
            logits = self.proj(z)                              # (B,K,N)

        if self.remap is not None:
            zeros  = torch.zeros_like(logits)
            logits = logits[:, self.used, :]

        # --- Soft/hard assignment ---------------------------------------- #
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)

        if self.remap is not None:
            zeros[:, self.used, :] = soft_one_hot
            soft_one_hot = zeros

        # --- Quantise ----------------------------------------------------- #
        z_q_cf = einsum(soft_one_hot, self.embed.weight, 'b k n, k d -> b d n')

        # KL divergence term
        qy   = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(
            qy * torch.log(qy * self.n_embed + 1e-10), dim=1
        ).mean()

        # Hard indices (B,N)
        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self._remap_to_used(ind)

        # --- Restore original layout for z_q ----------------------------- #
        if self.channels_last:               # want (B,N,D)
            z_q = z_q_cf.permute(0, 2, 1)
        else:                                # keep (B,D,N)
            z_q = z_q_cf
            # z_q = z_q.permute(0, 2, 1)

        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)

        return z_q, diff, ind

    # --------------------------------------------------------------------- #
    def get_codebook_entry(self, indices, shape):
        """
        indices : (B,N)      integer codes
        shape   : expected output shape before channels_first/last permute
                  • channels_last=True  → (B,N,D)
                  • channels_last=False → (B,D,N)
        """
        B, N = indices.shape
        if self.remap is not None:
            indices = self._unmap_to_all(indices)

        one_hot = F.one_hot(indices, num_classes=self.n_embed).float()  # (B,N,K)
        z_q     = einsum(one_hot, self.embed.weight, 'b n k, k d -> b n d')

        if self.channels_last:
            return z_q                      # (B,N,D)
        else:
            return z_q.permute(0, 2, 1)     # (B,D,N)
