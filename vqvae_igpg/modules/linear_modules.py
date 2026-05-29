import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
# ─────────────────────────────────────────────────────────────
# 1-D sinusoidal timestep embedding
# ─────────────────────────────────────────────────────────────
def get_timestep_embedding(
    timesteps: torch.Tensor,          # shape: (T,)
    embedding_dim: int
) -> torch.Tensor:
    """
    Create sinusoidal embeddings for a vector of diffusion timesteps.

    Works for any data dimensionality (1-D, 2-D, …) because the
    embedding depends only on the scalar timestep values.
    Returned shape: (T, embedding_dim)
    """
    assert timesteps.ndim == 1, "timesteps must be a 1-D tensor"

    half_dim = embedding_dim // 2
    freqs = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        * -(math.log(10000.0) / (half_dim - 1))
    )                                    # (half_dim,)

    args = timesteps.float()[:, None] * freqs[None, :]   # (T, half_dim)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (T, 2*half_dim)

    if embedding_dim % 2 == 1:           # zero-pad if odd
        emb = F.pad(emb, (0, 1))
    return emb

def swish(x, beta=1):
    return x * torch.sigmoid(beta * x)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
# ──────────────
# ─────────────────────────────────────────────────────────────
# Re-usable helper (same as in the encoder)
def _apply_linear(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    """
    Apply `layer` (C_in → C_out) independently at every 1-D position.
    """
    b, c, l = x.shape
    x = x.permute(0, 2, 1).reshape(-1, c)   # (B·L, C_in)
    x = layer(x)                            # (B·L, C_out)
    x = x.reshape(b, l, -1).permute(0, 2, 1)
    return x
# ─────────────────────────────────────────────────────────────


class UpsampleLinear1D(nn.Module):
    """
    (B, C, L) → (B, C, 2 L)
    • doubles the sequence length with nearest-neighbour repeat
    • optional per-position Linear(C→C) replaces the old 3-tap conv
    """
    def __init__(self, channels: int, with_linear: bool = True):
        super().__init__()
        self.with_linear = with_linear
        if with_linear:
            self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")      # (B, C, 2L)

        if self.with_linear:
            b, c, l = x.shape
            x = x.permute(0, 2, 1).reshape(-1, c)                  # (B·2L, C)
            x = self.proj(x)                                       # Linear
            x = x.reshape(b, l, c).permute(0, 2, 1)                # (B, C, 2L)

        return x


class DownsampleLinear1D(nn.Module):
    """
    (B, C, L) → (B, C, L⁄2)
    • optional Linear(C→C) first
    • 2-point average-pool halves the length (stride 2) – same spatial effect
      as a 1-D conv with kernel_size=3, stride=2 once padding is removed.
    """
    def __init__(self, channels: int, with_linear: bool = True):
        super().__init__()
        self.with_linear = with_linear
        if with_linear:
            self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_linear:
            b, c, l = x.shape
            x = x.permute(0, 2, 1).reshape(-1, c)                  # (B·L, C)
            x = self.proj(x)
            x = x.reshape(b, l, c).permute(0, 2, 1)                # (B, C, L)

        x = F.avg_pool1d(x, kernel_size=2, stride=2)               # (B, C, L/2)
        return x


class ResnetBlock1DLinear(nn.Module):
    """
    Residual block for 1-D feature maps that contains **no convolutions**.
    Every Linear layer is applied independently at every position along
    the length axis, i.e. it only mixes the *channel* dimension.

    Args
    ----
    in_channels      : #input channels
    out_channels     : #output channels (defaults to in_channels)
    dropout          : dropout prob. applied before the second projection
    temb_channels    : dimensionality of the timestep embedding; set to 0
                       to disable FiLM-style conditioning
    linear_shortcut  : if True and C_in ≠ C_out, use a learnable Linear
                       shortcut; otherwise identity.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float,
        temb_channels: int = 512,
        linear_shortcut: bool = True,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.has_shortcut = in_channels != out_channels

        # 1. Norm → Swish → Linear(C_in → C_out)
        self.norm1 = Normalize(in_channels)
        self.proj1 = nn.Linear(in_channels, out_channels)

        # 2. Optional FiLM from timestep embedding
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        # 3. Norm → Swish → Dropout → Linear(C_out → C_out)
        self.norm2  = Normalize(out_channels)
        self.drop   = nn.Dropout(dropout)
        self.proj2  = nn.Linear(out_channels, out_channels)

        # 4. Learnable shortcut if channel count changes
        if self.has_shortcut and linear_shortcut:
            self.shortcut = nn.Linear(in_channels, out_channels)

    # ---------------------------------------------------------
    def _apply_linear(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        """
        Helper: reshape (B, C, L) → (B·L, C) so nn.Linear acts per position,
        then restore to the original layout.
        """
        b, c, l = x.shape
        x = x.permute(0, 2, 1).reshape(-1, c)      # (B·L, C)
        x = layer(x)                               # Linear
        x = x.reshape(b, l, -1).permute(0, 2, 1)   # (B, C', L)
        return x
    # ---------------------------------------------------------

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h = swish(h)
        h = self._apply_linear(h, self.proj1)          # first projection

        if temb is not None:
            # FiLM-style affine bias, broadcast over length
            h = h + self.temb_proj(swish(temb))[:, :, None]

        h = self.norm2(h)
        h = swish(h)
        h = self.drop(h)
        h = self._apply_linear(h, self.proj2)          # second projection

        # --- residual connection ------------------------------------------
        if self.has_shortcut:
            if hasattr(self, "shortcut"):              # learnable Linear
                x = self._apply_linear(x, self.shortcut)
            # else: identity (channel counts already match)

        return x + h


class AttnBlock1DLinear(nn.Module):
    """
    Self-attention over a 1-D sequence with **no convolutions**.
    Every projection (q, k, v, out) is a Linear(C→C) applied
    independently at every position.

    The forward pass exactly mirrors the Conv2d version, just
    replacing (H·W) with L and Conv1×1 with Linear.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm     = Normalize(in_channels)          # (B, C, L) → (B, C, L)
        self.q_proj   = nn.Linear(in_channels, in_channels)
        self.k_proj   = nn.Linear(in_channels, in_channels)
        self.v_proj   = nn.Linear(in_channels, in_channels)
        self.proj_out = nn.Linear(in_channels, in_channels)

    # ------------------------------------------------------------------
    def _apply_linear(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        """
        Reshape so nn.Linear works per sequence element:
            (B, C, L) → (B·L, C) → Linear → (B, C, L)
        """
        b, c, l = x.shape
        x = x.permute(0, 2, 1).reshape(-1, c)           # (B·L, C)
        x = layer(x)
        x = x.reshape(b, l, -1).permute(0, 2, 1)        # (B, C, L)
        return x
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, L)
        returns residual output with same shape
        """
        h = self.norm(x)

        # Linear projections (per position)
        q = self._apply_linear(h, self.q_proj)          # (B, C, L)
        k = self._apply_linear(h, self.k_proj)
        v = self._apply_linear(h, self.v_proj)

        # --- scaled dot-product attention -----------------------------
        b, c, l = q.shape
        q = q.permute(0, 2, 1)        # (B, L, C)
        k = k                         # (B, C, L)
        attn = torch.bmm(q, k)        # (B, L, L)
        attn = attn * (c ** -0.5)
        attn = F.softmax(attn, dim=2) # softmax over key dimension

        # --- apply attention to values --------------------------------
        v  = v                        # (B, C, L)
        h  = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, L)

        # Output projection (per position)
        h = self._apply_linear(h, self.proj_out)

        # Residual connection
        return x + h




class Encoder1DLinear(nn.Module):
    """
    Linear-only analogue of the DDPM/LD-style encoder for 1-D inputs.

    * Input tensor: (B, my_channels, in_dim)
    * Output tensor: (B, 2·z_channels, L_out)  or  (B, z_channels, L_out)
      depending on `double_z`.

    The architecture mirrors the original 2-D ConvNet:
      fc_in  → proj_in  → [ResBlock/Attn, …, Downsample]×k
               ↓
             Middle (ResBlock-Attn-ResBlock)
               ↓
             norm + nonlin + proj_out
    All convolutions (3×3 or 1×1) are replaced with per-position Linear
    projections, and every spatial down-sampling stride-2 conv becomes
    an `avg_pool1d` in `DownsampleLinear1D`.
    """

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,  # if True, keep Linear after pool
        in_channels: int,
        resolution: int,                # 1-D “length” of the sequence
        z_channels: int,
        double_z: bool = True,
        in_dim: int = 2864,
        inreshape: bool = False,
        **ignore_kwargs,
    ):
        super().__init__()

        # -------- input reshape + first projection -------------------
        self.inreshape  = inreshape
        self.in_channels = in_channels
        self.in_dim     = in_dim
        self.resolution = resolution    # initial sequence length (L)

        self.fc_in   = nn.Linear(in_dim, resolution)            # (B, fch, in_dim) → (B, fch, fdim)
        self.proj_in = nn.Linear(in_channels, ch)         # per-pos Linear (C_in → ch)

        # -------- build the pyramid ---------------------------------
        self.temb_ch         = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks  = num_res_blocks

        curr_res   = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down  = nn.ModuleList()

        block_in = ch  # tracks channel dim through the network

        for i_level in range(self.num_resolutions):
            level = nn.Module()
            level.block = nn.ModuleList()
            level.attn  = nn.ModuleList()

            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                level.block.append(
                    ResnetBlock1DLinear(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    level.attn.append(AttnBlock1DLinear(block_in))

            # optional down-sampling (½ sequence length)
            if i_level != self.num_resolutions - 1:
                level.downsample = DownsampleLinear1D(
                    block_in, with_linear=resamp_with_conv
                )
                curr_res //= 2

            self.down.append(level)

        # -------- bottleneck ----------------------------------------
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1DLinear(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock1DLinear(block_in)
        self.mid.block_2 = ResnetBlock1DLinear(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # -------- output --------------------------------------------
        self.norm_out = Normalize(block_in)
        self.proj_out = nn.Linear(
            block_in, 2 * z_channels if double_z else z_channels
        )

    # ===============================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, ?, ?)  →  (B, *, L_out)
        """
        # 1. reshape raw input to (B, my_channels, in_dim)
        x = x.reshape(-1, self.in_channels, self.in_dim)  # may already be this shape

        # 2. per-channel fully-connected expansion
        x = self.fc_in(x)                         # (B, fch, fdim)

        #   now x.shape == (B, in_channels, resolution)

        # 4. first Linear “conv-in”
        h = _apply_linear(x, self.proj_in)        # (B, ch, L)
        hs = [h]

        # 5. down-sampling pyramid
        for lvl in range(self.num_resolutions):
            for blk_idx in range(self.num_res_blocks):
                h = self.down[lvl].block[blk_idx](hs[-1], temb=None)
                if len(self.down[lvl].attn) > 0:
                    h = self.down[lvl].attn[blk_idx](h)
                hs.append(h)

            if lvl != self.num_resolutions - 1:
                h = self.down[lvl].downsample(h)
                hs.append(h)

        # 6. bottleneck
        h = self.mid.block_1(h, temb=None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb=None)

        # 7. output head
        h = self.norm_out(h)
        h = swish(h)
        h = _apply_linear(h, self.proj_out)       # (B, z*2, L_out)

        return h



class Decoder1DLinear(nn.Module):
    """
    Convolution-free inverse of `Encoder1DLinear`.
    It expands a latent tensor  (B, z_channels, L_low)  back to the
    original vector shape, using only Linear layers plus nearest-
    neighbour 1-D up-sampling.
    """

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,          # final sequence length L_final
        z_channels: int,
        in_dim: int = 2864,
        give_pre_end: bool = False,
        inreshape: bool = True,
        **ignorekwargs,
    ):
        super().__init__()

        self.out_ch   = out_ch
        self.in_dim   = in_dim
        self.inreshape = inreshape
        self.give_pre_end = give_pre_end
        self.in_channels= in_channels
        self.resolution = resolution

        # ------------------------------------------------------
        # Compute lowest-resolution length and channels
        num_resolutions = len(ch_mult)
        block_in  = ch * ch_mult[-1]
        curr_res  = resolution // 2 ** (num_resolutions - 1)  # L_low
        self.z_shape = (1, z_channels, curr_res)

        # Linear "conv_in": z_channels → block_in
        self.proj_in = nn.Linear(z_channels, block_in)

        # ─── Bottleneck ───────────────────────────────────────
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1DLinear(
            in_channels=block_in, out_channels=block_in,
            dropout=dropout, temb_channels=0
        )
        self.mid.attn_1 = AttnBlock1DLinear(block_in)
        self.mid.block_2 = ResnetBlock1DLinear(
            in_channels=block_in, out_channels=block_in,
            dropout=dropout, temb_channels=0
        )

        # ─── Upsampling pyramid ───────────────────────────────
        self.up = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            level = nn.Module()
            level.block = nn.ModuleList()
            level.attn  = nn.ModuleList()

            block_out = ch * ch_mult[i_level]
            n_blocks  = num_res_blocks + 1  # mirror encoder

            for _ in range(n_blocks):
                level.block.append(
                    ResnetBlock1DLinear(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        temb_channels=0,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    level.attn.append(AttnBlock1DLinear(block_in))

            if i_level != 0:  # add up-sample except at top level
                level.upsample = UpsampleLinear1D(
                    block_in, with_linear=resamp_with_conv
                )
                curr_res *= 2

            # prepend to keep forward order intuitive
            self.up.insert(0, level)

        # ─── Output head ──────────────────────────────────────
        self.norm_out = Normalize(block_in)
        self.proj_out = nn.Linear(block_in, out_ch)

        # Final fully-connected to reach original dimensionality
        self.fc_out = nn.Linear(resolution, in_dim)

    # ==========================================================
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, z_channels, L_low)   →  reconstructed vector
        """
        # Linear "conv_in"
        h = _apply_linear(z, self.proj_in)

        # ─── Bottleneck ───────────────────────────────────────
        h = self.mid.block_1(h, temb=None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb=None)

        # ─── Upsampling pyramid ───────────────────────────────
        for i_level in reversed(range(len(self.up))):
            for idx in range(len(self.up[i_level].block)):
                h = self.up[i_level].block[idx](h, temb=None)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[idx](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h  # feature map before final norm / proj

        # ─── Output head ──────────────────────────────────────
        h = self.norm_out(h)
        h = swish(h)
        h = _apply_linear(h, self.proj_out)            # (B, out_ch, L_final)

        # Reshape to (B, fch, fdim) then reduce to in_dim with fc_out
        B, C, L = h.shape
        assert C * L == self.in_channels * self.resolution, "Dimension mismatch for reshape"
        h = h.view(B, self.in_channels, self.resolution)
        h = self.fc_out(h)                             # (B, fch, in_dim)
        if self.inreshape:
            h = h.view(B, self.in_channels * self.in_dim)      # flat vector

        return h
