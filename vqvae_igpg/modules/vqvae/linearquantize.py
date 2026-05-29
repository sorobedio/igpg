import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange

def norm_mse_chunk(pred, target, chunk_size=512, eps=1e-8):
    B, N = pred.shape
    T = N // chunk_size                       # assume padding to multiple
    pred   = pred.view(B, T, chunk_size)
    target = target.view(B, T, chunk_size)

    pred   = pred / (pred.norm(dim=2, keepdim=True) + eps)
    target = target / (target.norm(dim=2, keepdim=True) + eps)

    return (pred - target).pow(2).mean()      # scalar


class VectorQuantizer1D(nn.Module):
    """
    Discretisation bottleneck for 1-D latent sequences.

    Args
    ----
    n_e   : number of codebook embeddings
    e_dim : embedding dimension (must match channel count C)
    beta  : commitment cost (see VQ-VAE paper)
    """

    def __init__(self, n_e: int, e_dim: int, beta: float):
        super().__init__()
        self.n_e  = n_e
        self.e_dim = e_dim
        self.beta  = beta

        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    # ---------------------------------------------------------
    def forward(self, z: torch.Tensor):
        """
        z : (B, C, L)   — continuous latent from encoder

        Returns
        -------
        z_q          : (B, C, L)  — quantised latent
        loss         : commitment + codebook loss
        info_tuple   : (perplexity, one-hots, indices)
        """
        # --- flatten along sequence length -------------------
        z_perm = z.permute(0, 2, 1).contiguous()          # (B, L, C)
        z_flat = z_perm.view(-1, self.e_dim)              # (B·L, C)

        # --- compute distances to codebook ------------------
        dist = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # --- nearest neighbour lookup -----------------------
        indices = torch.argmin(dist, dim=1)               # (B·L,)
        z_q = self.embedding(indices).view(z_perm.shape)  # (B, L, C)

        # --- losses -----------------------------------------
        z_q_perm = z_q                                    # (B, L, C)
        z_q_flat = z_q_perm.view_as(z_flat)

        commitment = torch.mean((z_q_flat.detach() - z_flat) ** 2)
        codebook   = torch.mean((z_q_flat - z_flat.detach()) ** 2)
        loss = commitment + self.beta * codebook

        # --- gradient passthrough (straight-through) --------
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_flat.view_as(z_perm).permute(0, 2, 1)    # back to (B, C, L)

        # --- perplexity -------------------------------------
        one_hot = F.one_hot(indices, self.n_e).type(z.dtype)
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, loss, (perplexity, one_hot, indices)

    # ---------------------------------------------------------
    def get_codebook_entry(self, indices: torch.Tensor, shape=None):
        """
        Replicates the original VQ-VAE helper for 1-D data.

        Parameters
        ----------
        indices : LongTensor
            • flat (N,)  or shaped (B·L,) index list produced by argmin.
        shape   : tuple | None
            When not None, expected to be **(batch, length, channel)**,
            i.e. (B, L, C).  The returned tensor is then permuted to the
            channel-first layout (B, C, L).  When shape is None you get the
            flat (N, C) tensor exactly like the 2-D version did.

        Returns
        -------
        z_q : • (B, C, L)  if shape is given
              • (N, C)     if shape is None
        """
        # ---------- step 1: build one-hot matrix ------------------
        n_codes = indices.shape[0]
        min_encodings = torch.zeros(n_codes, self.n_e, device=indices.device)
        min_encodings.scatter_(1, indices[:, None], 1)            # (N, n_e)

        # ---------- step 2: matrix-multiply with codebook --------
        z_q = torch.matmul(min_encodings, self.embedding.weight)  # (N, C)

        # ---------- step 3: reshape & permute to (B, C, L) -------
        if shape is not None:
            # shape is (B, L, C)
            z_q = z_q.view(shape)          # → (B, L, C)
            z_q = z_q.permute(0, 2, 1)     # → (B, C, L)

        return z_q

# ─────────────────────────────────────────────────────────────
def _apply_linear(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    """(B, C_in, L)  →  (B, C_out, L)  with a position-wise Linear."""
    b, c, l = x.shape
    x = x.permute(0, 2, 1).reshape(-1, c)   # (B·L, C_in)
    x = layer(x)                            # (B·L, C_out)
    return x.view(b, l, -1).permute(0, 2, 1)
# ─────────────────────────────────────────────────────────────


class GumbelQuantize1D(nn.Module):
    """
    Gumbel-Softmax VQ layer for 1-D latent tensors (B, C, L).

    * `proj` turns channels → `n_embed` logits at every position
    * `embed` stores the codebook (n_embed, embedding_dim)
    """

    def __init__(
        self,
        num_hiddens: int,
        embedding_dim: int,
        n_embed: int,
        straight_through: bool = True,
        kl_weight: float = 5e-4,
        temp_init: float = 1.0,
        use_vqinterface: bool = True,
        remap: str | None = None,
        unknown_index: str | int = "random",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.use_vqinterface = use_vqinterface

        # 1×1 “conv” → Linear
        self.proj = nn.Linear(num_hiddens, n_embed)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        # ---------------- optional index-remap -----------------
        self.remap = remap
        if remap is not None:
            self.register_buffer("used", torch.tensor(np.load(remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed += 1
            print(
                f"Remapping {n_embed} indices → {self.re_embed}. "
                f"Unknown ↦ {self.unknown_index}."
            )
        else:
            self.re_embed = n_embed

    # ─────────────────────────────────────────────────────────
    # helpers for remap mode
    def remap_to_used(self, inds):  # inds: (B, L)
        ishape = inds.shape
        inds = inds.view(inds.size(0), -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, :]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, new[unknown].shape, device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.view(ishape)

    def unmap_to_all(self, inds):  # inds: (B, L)
        ishape = inds.shape
        inds = inds.view(inds.size(0), -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:          # “extra” token
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None].expand(inds.size(0), -1), 1, inds)
        return back.view(ishape)
    # ─────────────────────────────────────────────────────────

    # =========================================================
    def forward(self, z: torch.Tensor, temp: float | None = None, return_logits=False):
        """
        z : (B, C, L) continuous latent
        """
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = _apply_linear(z, self.proj)            # (B, n_embed, L)

        # ---------- remap option ------------------------------
        if self.remap is not None:
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, :]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)  # (B, n_embed, L)

        if self.remap is not None:
            full_zeros[:, self.used, :] = soft_one_hot
            soft_one_hot = full_zeros

        # (B, embedding_dim, L)
        z_q = einsum("b n l, n d -> b d l", soft_one_hot, self.embed.weight)

        # -------- KL divergence to uniform prior --------------
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        # -------- hard indices (B, L) --------------------------
        ind = soft_one_hot.argmax(dim=1)                # (B, L)
        if self.remap is not None:
            ind = self.remap_to_used(ind)

        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    # =========================================================
    def get_codebook_entry(self, indices: torch.Tensor, shape):
        """
        Reconstruct embeddings by index.

        Parameters
        ----------
        indices : (B*L,) **flattened** index list  – same convention
                  as the original implementation.
        shape   : (B, L, C)  target shape *before* channel-first permute

        Returns
        -------
        z_q : (B, C, L)  quantised tensor
        """
        b, l, c = shape
        assert b * l == indices.numel(), "shape mismatch with indices"

        # reshape flat → (B, L)
        indices = indices.view(b, l)

        if self.remap is not None:
            indices = self.unmap_to_all(indices)

        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 2, 1).float()  # (B, n_embed, L)
        z_q = einsum("b n l, n d -> b d l", one_hot, self.embed.weight)                 # (B, C, L)
        return z_q


class VectorQuantizer2_1D(nn.Module):
    """
    1-D version of VectorQuantizer2.
    Expects latents shaped (B, C, L) and returns the same shape.
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float,
        remap: str | None = None,
        unknown_index: str | int = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ):
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.sane_index_shape = sane_index_shape

        # codebook
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        # ---- optional remap ----------------------------------
        self.remap = remap
        if remap is not None:
            self.register_buffer("used", torch.tensor(np.load(remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed += 1
            print(
                f"Remapping {n_e} indices → {self.re_embed}. "
                f"Unknown ↦ {self.unknown_index}."
            )
        else:
            self.re_embed = n_e

    # ─────────────────────────────────────────────────────────
    # remap helpers (unchanged except for shape semantics)
    def remap_to_used(self, inds):
        ishape = inds.shape
        inds = inds.reshape(ishape[0], -1)                # (B, L)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, new[unknown].shape, device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:            # “extra” token
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None].expand(inds.size(0), -1), 1, inds)
        return back.reshape(ishape)
    # ─────────────────────────────────────────────────────────

    # =========================================================
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        # Interface-compatibility assertions (kept from original)
        assert temp is None or temp == 1.0
        assert rescale_logits is False
        assert return_logits is False

        # -------- reshape to (B, L, C) and flatten -----------
        z_perm = rearrange(z, "b c l -> b l c").contiguous()
        z_flat = z_perm.view(-1, self.e_dim)              # (B·L, C)

        # -------- compute distances --------------------------
        d = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flat, rearrange(self.embedding.weight, "n d -> d n"))
        )

        # -------- nearest neighbour --------------------------
        min_enc_idx = torch.argmin(d, dim=1)              # (B·L,)
        z_q_flat = self.embedding(min_enc_idx)            # (B·L, C)

        # -------- loss --------------------------------------
        if not self.legacy:
            loss = self.beta * torch.mean((z_q_flat.detach() - z_flat) ** 2) + \
                   torch.mean((z_q_flat - z_flat.detach()) ** 2)
        else:
            loss = torch.mean((z_q_flat.detach() - z_flat) ** 2) + \
                   self.beta * torch.mean((z_q_flat - z_flat.detach()) ** 2)

        # -------- straight-through estimator ---------------
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        # -------- reshape back to (B, C, L) ----------------
        z_q = z_q_flat.view_as(z_perm).permute(0, 2, 1).contiguous()

        # -------- optional remap ----------------------------
        if self.remap is not None:
            min_enc_idx = min_enc_idx.view(z.shape[0], -1)
            min_enc_idx = self.remap_to_used(min_enc_idx)
            min_enc_idx = min_enc_idx.view(z.shape[0], -1)

        if self.sane_index_shape:
            min_enc_idx = min_enc_idx.view(z_q.shape[0], z_q.shape[2])  # (B, L)

        return z_q, loss, (None, None, min_enc_idx)

    # =========================================================
    def get_codebook_entry(self, indices, shape):
        """
        indices : flat 1-D tensor of length B·L  (same as argmin output)
        shape   : (B, L, C)   – desired (before permute)
        """
        if self.remap is not None:
            indices = indices.view(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.view(-1)

        # lookup
        z_q = self.embedding(indices)                     # (B·L, C)

        if shape is not None:
            z_q = z_q.view(shape)                         # (B, L, C)
            z_q = z_q.permute(0, 2, 1).contiguous()       # (B, C, L)

        return z_q




class EmbeddingEMA1D(nn.Module):
    """
    Exponential-moving-average (EMA) code-book, ready for 1-D latents.

    • `weight`        : (num_tokens, codebook_dim) – the embedding table
    • `cluster_size`  : running sum of soft‐assignment counts
    • `embed_avg`     : running sum of latent vectors associated with tokens
    • `forward()`     : standard `F.embedding` lookup; accepts arbitrary
                        index shapes, e.g. (B, L) or (B·L,)
    • `cluster_size_ema_update()` & `embed_avg_ema_update()` :
                        accumulate new statistics (call from your quantiser)
    • `weight_update()` : normalise and copy EMA state into `weight`
    """

    def __init__(
        self,
        num_tokens: int,
        codebook_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.decay = decay
        self.eps = eps

        weight = torch.randn(num_tokens, codebook_dim)
        self.weight       = nn.Parameter(weight,         requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg    = nn.Parameter(weight.clone(), requires_grad=False)

        # flag used by some projects to disable EMA during the first steps
        self.update = True

    # ---------------------------------------------------------
    def forward(self, embed_id: torch.Tensor) -> torch.Tensor:
        """
        embed_id : *(any shape)* integer tensor of code indices
        returns  : (..., codebook_dim)
        """
        return F.embedding(embed_id, self.weight)

    # ---------------------------------------------------------
    def cluster_size_ema_update(self, new_cluster_size: torch.Tensor):
        """
        new_cluster_size : (num_tokens,) vector of counts from the current mini-batch
        """
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1.0 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg: torch.Tensor):
        """
        new_embed_avg : (num_tokens, codebook_dim) sum of latents assigned to each token
        """
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1.0 - self.decay)

    def weight_update(self):
        """
        Call periodically (or every step) **after** the two EMA accumulators
        have been updated with the current batch’s statistics.
        """
        n = self.cluster_size.sum()
        num_tokens = self.cluster_size.size(0)

        # Laplace smoothing
        smoothed_cs = (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n

        # Normalise average embedding by (smoothed) cluster size
        embed_normalised = self.embed_avg / smoothed_cs.unsqueeze(1)

        # Copy into actual code-book
        self.weight.data.copy_(embed_normalised)

    # ---------------------------------------------------------
    # Optional convenience: accumulate stats from flat indices
    # ---------------------------------------------------------
    @torch.no_grad()
    def accumulate_from_indices(self, indices: torch.Tensor, latents: torch.Tensor):
        """
        Helper when your quantiser gives you per-position indices and latents.

        indices : (B·L,)   long tensor
        latents : (B·L, D) float tensor (detached latent vectors)

        Computes per-token counts and sums, then applies EMA updates.
        """
        device = indices.device
        num_tokens = self.weight.size(0)
        D = self.weight.size(1)

        # one-hot count & sum
        one_hot = F.one_hot(indices, num_tokens).type_as(latents)           # (N, num_tokens)
        new_cluster_size = one_hot.sum(0)                                   # (num_tokens,)
        new_embed_avg = one_hot.T @ latents                                 # (num_tokens, D)

        # EMA updates
        self.cluster_size_ema_update(new_cluster_size)
        self.embed_avg_ema_update(new_embed_avg)
        self.weight_update()






# ----------------------------------------------------------------------
# Exponential-moving-average code-book from the previous message
# (kept identical – just import or paste in the same file)
# ----------------------------------------------------------------------
class EmbeddingEMA1D(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight       = nn.Parameter(weight,         requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg    = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, idx):
        return F.embedding(idx, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cs = (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        self.weight.data.copy_(self.embed_avg / smoothed_cs.unsqueeze(1))


# ----------------------------------------------------------------------
# EMA Vector Quantiser adapted for 1-D latents
# ----------------------------------------------------------------------
class EMAVectorQuantizer1D(nn.Module):
    """
    Exponential-moving-average variant of VectorQuantiser for 1-D data.

    • Input  / output: (B, C, L)
    • `n_embed`       : size of the code-book
    • `embedding_dim` : latent channel dimension  C
    """

    def __init__(
        self,
        n_embed: int,
        embedding_dim: int,
        beta: float,
        decay: float = 0.99,
        eps: float = 1e-5,
        remap: str | None = None,
        unknown_index: str | int = "random",
    ):
        super().__init__()

        self.num_tokens   = n_embed
        self.codebook_dim = embedding_dim
        self.beta         = beta

        self.embedding = EmbeddingEMA1D(n_embed, embedding_dim, decay, eps)

        # ------------- optional remap logic -----------------
        self.remap = remap
        if remap is not None:
            self.register_buffer("used", torch.tensor(np.load(remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed += 1
            print(
                f"Remapping {n_embed} indices → {self.re_embed}. "
                f"Unknown ↦ {self.unknown_index}."
            )
        else:
            self.re_embed = n_embed

    # ---------------- remap helpers (unchanged) -------------
    def remap_to_used(self, inds):
        ishape = inds.shape
        inds = inds.reshape(inds.size(0), -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, new[unknown].shape, device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        inds = inds.reshape(inds.size(0), -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None].expand(inds.size(0), -1), 1, inds)
        return back.reshape(ishape)

    # ========================================================
    def forward(self, z: torch.Tensor):
        """
        z : (B, C, L) continuous latent
        """
        # ---- reshape to (B, L, C) then flatten -------------
        z_perm = rearrange(z, "b c l -> b l c")
        z_flat = z_perm.reshape(-1, self.codebook_dim)

        # ---- distance to code-book -------------------------
        d = (
            z_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2 * torch.einsum("bd,nd->bn", z_flat, self.embedding.weight)
        )

        encoding_indices = torch.argmin(d, dim=1)                # (B·L,)
        z_q_flat = self.embedding(encoding_indices)              # (B·L, C)

        # ---- perplexity ------------------------------------
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z_flat.dtype)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # ---- EMA updates -----------------------------------
        if self.training and self.embedding.update:
            self.embedding.cluster_size_ema_update(encodings.sum(0))
            embed_sum = encodings.T @ z_flat                     # (num_tokens, C)
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)

        # ---- commitment loss -------------------------------
        loss = self.beta * F.mse_loss(z_q_flat.detach(), z_flat)

        # ---- straight-through estimator --------------------
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_flat.view_as(z_perm).permute(0, 2, 1).contiguous()  # (B, C, L)

        return z_q, loss, (perplexity, encodings, encoding_indices)

    # ========================================================
    def get_codebook_entry(self, indices: torch.Tensor, shape):
        """
        indices : flat (B·L,) tensor
        shape   : (B, L, C)  target shape before permuting

        returns : (B, C, L)
        """
        if self.remap is not None:
            indices = indices.view(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.view(-1)

        z_q = self.embedding(indices)                      # (B·L, C)
        z_q = z_q.view(shape)                              # (B, L, C)
        return z_q.permute(0, 2, 1).contiguous()           # (B, C, L)
