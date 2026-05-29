import math
from math import pi
from functools import wraps
from dataclasses import dataclass

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# -----------------
# small utilities
# -----------------
def exists(x): return x is not None
def default(x, d): return x if exists(x) else d

def cache_fn(f):
    cache = {}
    @wraps(f)
    def cached(*args, _cache=True, key=None, **kwargs):
        if not _cache: return f(*args, **kwargs)
        nonlocal cache
        if key in cache: return cache[key]
        out = f(*args, **kwargs)
        cache[key] = out
        return out
    return cached

def fourier_encode(x, max_freq, num_bands=4):
    """
    x: [..., 1] or scalar positions in [-1, 1]
    returns [..., 2*num_bands + 1]   (sin/cos pairs + raw x)
    """
    x = x.unsqueeze(-1)                # [..., 1] -> [..., 1, 1]
    device, dtype, orig = x.device, x.dtype, x
    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(None,) * (x.ndim - 1) + (Ellipsis,)]  # broadcast to x
    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    return torch.cat([x, orig], dim=-1)

# -----------------
# blocks
# -----------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context) and 'context' in kwargs and exists(kwargs['context']):
            kwargs = {**kwargs, 'context': self.norm_context(kwargs['context'])}
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner = heads * dim_head
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q  = nn.Linear(query_dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, query_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)                         # [b, nq, inner]
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            # mask is for keys (context tokens)
            mask = rearrange(mask, 'b n -> b 1 n')
            mask = repeat(mask, 'b 1 n -> (b h) 1 n', h=h)
            max_neg = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg)
        attn = self.drop(sim.softmax(dim=-1))
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

# -----------------
# chunking helpers
# -----------------
def chunkify(x: torch.Tensor, chunk_size: int):
    """
    x: [B, L]  (flat weight vectors)
    returns:
      chunks: [B, N, C] with C=chunk_size, N = ceil(L / C)
      mask_tokens: [B, N] bool (valid tokens)
      mask_elems:  [B, N, C] bool (valid elements in each chunk)
      orig_len
    """
    B, L = x.shape
    C = chunk_size
    N = (L + C - 1) // C
    pad_len = N * C - L
    if pad_len > 0:
        x_padded = F.pad(x, (0, pad_len))
    else:
        x_padded = x
    chunks = rearrange(x_padded, 'b (n c) -> b n c', n=N, c=C)
    # masks
    mask_tokens = torch.ones(B, N, dtype=torch.bool, device=x.device)
    if pad_len > 0:
        # last chunk has some invalid tail
        mask_elems = torch.ones(B, N, C, dtype=torch.bool, device=x.device)
        mask_elems[:, -1, -pad_len:] = False
    else:
        mask_elems = torch.ones(B, N, C, dtype=torch.bool, device=x.device)
    return chunks, mask_tokens, mask_elems, L

def unchunkify(chunks: torch.Tensor, orig_len: int):
    """
    chunks: [B, N, C]
    returns: [B, orig_len]
    """
    x = rearrange(chunks, 'b n c -> b (n c)')
    return x[:, :orig_len]

# -----------------
# Perceiver VAE
# -----------------
@dataclass
class PerceiverVAEConfig:
    # model sizes
    input_channels: int = 0            # we use pure numeric chunks, so 0 extra channels
    chunk_size: int = 1024             # C: features per token (one "token" is one chunk)
    num_freq_bands: int = 6
    max_freq: float = 10.0
    latent_dim: int = 512
    num_latents: int = 512
    depth: int = 6
    self_per_cross_attn: int = 1
    cross_heads: int = 1
    latent_heads: int = 8
    cross_dim_head: int = 64
    latent_dim_head: int = 64
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    weight_tie_layers: bool = False

    # VAE
    z_dim: int = 256

    # decoder
    dec_depth: int = 4
    dec_self_per_cross_attn: int = 1

    # loss
    recon_loss: str = "mse"  # "mse" or "l1"

class PerceiverChunkVAE(nn.Module):
    """
    Perceiver-style VAE for reconstructing large weight vectors by chunking.

    Encoder:
      - Tokens = chunks (size C), + Fourier position features for token index.
      - Latent array (num_latents, latent_dim) cross-attends to tokens, then latent self-attn (repeated).
      - Mean-pool latents -> μ, logσ² in R^{z_dim}.

    Decoder:
      - Build output queries for each chunk index (positionally encoded and projected).
      - Queries cross-attend to a context derived from z (broadcast as length-1 seq).
      - Project queries to chunk_size to reconstruct each chunk.

    Forward returns:
      recon: [B, L]   (trimmed to original length)
      kld:   scalar
      recon_loss: scalar
      aux dict with shapes / masks
    """
    def __init__(self, cfg: PerceiverVAEConfig):
        super().__init__()
        self.cfg = cfg

        # ------- input feature dimension (chunk values + position features) -------
        # input_axis = 1 (sequence of chunks). Position features are for token index only.
        pos_feat = (2 * cfg.num_freq_bands + 1) * 1  # 1 axis (token index)
        input_dim = cfg.chunk_size + pos_feat

        # ------- encoder latent array -------
        self.latents = nn.Parameter(torch.randn(cfg.num_latents, cfg.latent_dim))

        # factories (optionally tied)
        get_cross_attn = lambda: PreNorm(cfg.latent_dim,
            Attention(cfg.latent_dim, input_dim, heads=cfg.cross_heads,
                      dim_head=cfg.cross_dim_head, dropout=cfg.attn_dropout),
            context_dim=input_dim
        )
        get_cross_ff  = lambda: PreNorm(cfg.latent_dim, FeedForward(cfg.latent_dim, dropout=cfg.ff_dropout))
        get_latent_attn = lambda: PreNorm(cfg.latent_dim,
            Attention(cfg.latent_dim, heads=cfg.latent_heads,
                      dim_head=cfg.latent_dim_head, dropout=cfg.attn_dropout)
        )
        get_latent_ff = lambda: PreNorm(cfg.latent_dim, FeedForward(cfg.latent_dim, dropout=cfg.ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn,
            (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.enc_layers = nn.ModuleList([])
        for i in range(cfg.depth):
            should_cache = i > 0 and cfg.weight_tie_layers
            cache_args = {'_cache': should_cache}
            self_attns = nn.ModuleList([
                nn.ModuleList([
                    get_latent_attn(**cache_args, key=k),
                    get_latent_ff(**cache_args, key=k),
                ]) for k in range(cfg.self_per_cross_attn)
            ])
            self.enc_layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.enc_out_norm = nn.LayerNorm(cfg.latent_dim)
        self.to_mu     = nn.Linear(cfg.latent_dim, cfg.z_dim)
        self.to_logvar = nn.Linear(cfg.latent_dim, cfg.z_dim)

        # ------- decoder -------
        # Make queries from positional encodings of chunk indices, projected to latent_dim
        self.query_pos_proj = nn.Linear((2 * cfg.num_freq_bands + 1), cfg.latent_dim)

        # z context projector -> produces context_dim for cross-attn keys/values
        self.z_to_context = nn.Linear(cfg.z_dim, cfg.latent_dim)

        # decoder layers: queries are the "x", context = z-context (length 1)
        get_dec_cross_attn = lambda: PreNorm(cfg.latent_dim,
            Attention(cfg.latent_dim, cfg.latent_dim, heads=cfg.cross_heads,
                      dim_head=cfg.cross_dim_head, dropout=cfg.attn_dropout),
            context_dim=cfg.latent_dim
        )
        get_dec_cross_ff  = lambda: PreNorm(cfg.latent_dim, FeedForward(cfg.latent_dim, dropout=cfg.ff_dropout))
        get_dec_self_attn = lambda: PreNorm(cfg.latent_dim,
            Attention(cfg.latent_dim, heads=cfg.latent_heads,
                      dim_head=cfg.latent_dim_head, dropout=cfg.attn_dropout)
        )
        get_dec_self_ff   = lambda: PreNorm(cfg.latent_dim, FeedForward(cfg.latent_dim, dropout=cfg.ff_dropout))

        get_dec_cross_attn, get_dec_cross_ff, get_dec_self_attn, get_dec_self_ff = map(cache_fn,
            (get_dec_cross_attn, get_dec_cross_ff, get_dec_self_attn, get_dec_self_ff))

        self.dec_layers = nn.ModuleList([])
        for i in range(cfg.dec_depth):
            should_cache = i > 0 and cfg.weight_tie_layers
            cache_args = {'_cache': should_cache}
            self_attns = nn.ModuleList([
                nn.ModuleList([
                    get_dec_self_attn(**cache_args, key=k),
                    get_dec_self_ff(**cache_args, key=k),
                ]) for k in range(cfg.dec_self_per_cross_attn)
            ])
            self.dec_layers.append(nn.ModuleList([
                get_dec_cross_attn(**cache_args),
                get_dec_cross_ff(**cache_args),
                self_attns
            ]))

        # final projection per query token -> chunk_size
        self.to_chunk = nn.Linear(cfg.latent_dim, cfg.chunk_size)

        # loss
        if cfg.recon_loss.lower() == "mse":
            self._recon_loss = lambda pred, tgt, mask: ((pred - tgt) ** 2) * mask
        elif cfg.recon_loss.lower() == "l1":
            self._recon_loss = lambda pred, tgt, mask: (pred - tgt).abs() * mask
        else:
            raise ValueError("recon_loss must be 'mse' or 'l1'")

    # ----- positional queries for a sequence length N -----
    def build_token_positions(self, N: int, device, dtype):
        # positions in [-1,1], shape [N]
        pos = torch.linspace(-1., 1., steps=N, device=device, dtype=dtype)
        enc = fourier_encode(pos, self.cfg.max_freq, self.cfg.num_freq_bands)  # [N, 2F+1]
        q = self.query_pos_proj(enc)                                           # [N, latent_dim]
        return q

    # ----- encoder -----
    def encode(self, chunks: torch.Tensor, mask_tokens: torch.Tensor):
        """
        chunks: [B, N, C]
        mask_tokens: [B, N]  (True for valid tokens)
        returns: mu, logvar, latents_out
        """
        B, N, C = chunks.shape
        device, dtype = chunks.device, chunks.dtype

        # build token features: [value chunk | position features]
        # position features per token index
        pos = torch.linspace(-1., 1., steps=N, device=device, dtype=dtype)     # [N]
        pos_enc = fourier_encode(pos, self.cfg.max_freq, self.cfg.num_freq_bands)  # [N, 2F+1]
        pos_enc = repeat(pos_enc, 'n d -> b n d', b=B)

        tokens = torch.cat([chunks, pos_enc], dim=-1)   # [B, N, C + pos]
        x = repeat(self.latents, 'n d -> b n d', b=B)   # [B, num_latents, latent_dim]

        # cross + self stacks
        for cross_attn, cross_ff, self_attns in self.enc_layers:
            x = x + cross_attn(x, context=tokens, mask=mask_tokens)
            x = x + cross_ff(x)
            for self_attn, self_ff in self_attns:
                x = x + self_attn(x)
                x = x + self_ff(x)

        x = self.enc_out_norm(x)                        # [B, num_latents, latent_dim]
        pooled = x.mean(dim=1)                          # [B, latent_dim]
        mu = self.to_mu(pooled)                         # [B, z_dim]
        logvar = self.to_logvar(pooled)                 # [B, z_dim]
        return mu, logvar, x

    # ----- reparameterization -----
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ----- decoder -----
    def decode(self, z: torch.Tensor, num_chunks: int):
        """
        z: [B, z_dim]
        returns: pred_chunks [B, N, C]
        """
        B, _ = z.shape
        device, dtype = z.device, z.dtype

        # output queries from chunk positions
        queries = self.build_token_positions(num_chunks, device, dtype)  # [N, latent_dim]
        queries = repeat(queries, 'n d -> b n d', b=B)                   # [B, N, latent_dim]

        # make a length-1 context from z (projected)
        ctx = self.z_to_context(z)                                      # [B, latent_dim]
        ctx = rearrange(ctx, 'b d -> b 1 d')                            # [B, 1, latent_dim]

        # run decoder stacks
        x = queries
        for cross_attn, cross_ff, self_attns in self.dec_layers:
            x = x + cross_attn(x, context=ctx)   # attend to z-context
            x = x + cross_ff(x)
            for self_attn, self_ff in self_attns:
                x = x + self_attn(x)             # optional mixing across chunk queries
                x = x + self_ff(x)

        pred_chunks = self.to_chunk(x)           # [B, N, chunk_size]
        return pred_chunks

    # ----- end-to-end -----
    def forward(self, weights_flat: torch.Tensor):
        """
        weights_flat: [B, L] raw weight vectors
        Returns:
          recon: [B, L]
          kld: scalar
          rec_loss: scalar
          info: dict with shapes/masks
        """
        chunks, mask_tokens, mask_elems, orig_len = chunkify(weights_flat, self.cfg.chunk_size)   # [B,N,C], [B,N], [B,N,C]
        mu, logvar, _ = self.encode(chunks, mask_tokens)
        z = self.reparameterize(mu, logvar)
        pred_chunks = self.decode(z, num_chunks=chunks.shape[1])                                  # [B,N,C]

        # reconstruction loss on valid elements only
        elem_mask = mask_elems.float()                                                            # [B,N,C]
        rec = self._recon_loss(pred_chunks, chunks, elem_mask).sum() / elem_mask.sum().clamp_min(1)

        # KL (sum over dim, mean over batch)
        kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=-1).mean()

        # stitch back and trim padding
        recon_full = unchunkify(pred_chunks, orig_len)                                            # [B, L]

        return recon_full, kld, rec, {
            "mu": mu, "logvar": logvar, "z": z,
            "orig_len": orig_len, "num_chunks": chunks.shape[1],
            "chunk_size": self.cfg.chunk_size
        }

cfg = PerceiverVAEConfig(
    chunk_size=2048,          # pick a chunk granularity that fits your memory
    num_freq_bands=8,
    max_freq=16.0,
    latent_dim=512,
    num_latents=512,
    depth=6,
    z_dim=256,
    dec_depth=4,
    recon_loss="mse",
)

model = PerceiverChunkVAE(cfg)

B, L = 4, 12_345_678               # e.g., a big flattened weight vector
x = torch.randn(B, L)

recon, kld, rec, info = model(x)
loss = rec + 1e-4 * kld            # scale KL as you like (β-VAE, KL warmup, etc.)
loss.backward()