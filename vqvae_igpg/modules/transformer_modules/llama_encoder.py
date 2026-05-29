import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

#############################
# Weight Embedding & Decoding
#############################

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

class MLPResidualBlock(nn.Module):
    def __init__(
        self, input_size, hidden_size,  pre_layer_norm=True, post_dropout=False
    ):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, input_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)

class WeightEmbed(nn.Module):
    """
    Embeds a flattened weight vector by dividing it into fixed-size chunks (tokens)
    and projecting each chunk into an embedding dimension. If the vector length
    is not divisible by the chunk size, the input is padded with zeros.
    """
    def __init__(self, chunk_size: int, embed_dim: int, conv: bool = True, flatten: bool = True):
        super().__init__()
        self.conv = conv
        self.flatten = flatten
        self.chunk_size = chunk_size

        if conv:
            # Input x: (B, L) → unsqueeze to (B, 1, L)
            # Conv1d outputs shape (B, embed_dim, num_tokens) with num_tokens = ceil(L/chunk_size)
            self.proj = nn.Conv1d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=chunk_size,
                stride=chunk_size
            )
        else:
            # Reshape x into (B, num_tokens, chunk_size) then apply a tokenwise linear projection.
            self.proj = nn.Linear(chunk_size, embed_dim)

        # self.proj= MLPResidualBlock(chunk_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        remainder = L % self.chunk_size
        if remainder != 0:
            pad_length = self.chunk_size - remainder
            # Pad the last dimension so that L becomes a multiple of chunk_size.
            x = F.pad(x, (0, pad_length), mode='constant', value=0)
        if self.conv:
            # Add channel dimension: (B, 1, L)
            x = x.unsqueeze(1)
            # Conv1d: output shape (B, embed_dim, num_tokens)
            x = self.proj(x)
            # Permute to (B, num_tokens, embed_dim)
            x = x.transpose(1, 2)
        else:
            # Now x has shape (B, L_padded) with L_padded divisible by chunk_size.
            num_tokens = x.shape[1] // self.chunk_size
            x = x.view(B, num_tokens, self.chunk_size)
            x = self.proj(x)
        # x = self.mlp(x)
        return x  # (B, num_tokens, embed_dim)


class WeightDecode(nn.Module):
    """
    Decodes embedded weight tokens back to the original weight vector.
    """
    def __init__(self, chunk_size: int, embed_dim: int, out_channels: int = 1, conv: bool = True):
        super().__init__()
        self.conv = conv
        self.chunk_size = chunk_size
        self.out_channels = out_channels

        if conv:
            # Inverse of Conv1d: use 1D transposed convolution.
            self.proj = nn.ConvTranspose1d(
                in_channels=embed_dim,
                out_channels=out_channels,
                kernel_size=chunk_size,
                stride=chunk_size
            )
        else:
            # Use a linear layer to map each token back to a chunk.
            self.proj = nn.Linear(embed_dim, chunk_size)
        # self.mlp= MLPResidualBlock(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, original_length: int = None) -> torch.Tensor:
        # x is expected to be of shape (B, num_tokens, embed_dim)
        # x = self.mlp(x)
        if self.conv:
            # Rearrange to (B, embed_dim, num_tokens)
            x = x.transpose(1, 2)
            x = self.proj(x)  # (B, out_channels, L) with L = num_tokens * chunk_size
            if self.out_channels == 1:
                x = x.squeeze(1)   # (B, L)
        else:
            # print(x.shape)
            # print(self.proj)
            x = self.proj(x)       # (B, num_tokens, chunk_size)

        x = x.view(x.shape[0], -1)  # (B, num_tokens * chunk_size)
        if original_length is not None and x.shape[1] > original_length:
            # Trim the padded extra elements
            x = x[:, :original_length]

        return x

#############################
# Transformer Modules & RoPE Utilities
#############################

class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dtype: torch.dtype):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False)
        # self.activation = F.silu  # Using SiLU activation
        # self.activation =nn.Tanh()
        # self.activation = nn.LeakyReLU()
        # self.activation = nn.ReLU()
        self.activation = GELU()
        # self.activation =  torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x_fc1)
        x = self.activation(x_fc1) * x_fc2
        return  self.fc3(x)

        # x_fc1 = self.fc1(x)
        # x = self.activation(x_fc1)
        # return self.fc3(x)

def precompute_rope_params(
        head_dim: int,
        theta_base: int = 10_000,
        context_length: int = 4096,
        original_context_length: int = None,
        low_freq_factor: float = None,
        high_freq_factor: float = None,
        factor: float = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    if original_context_length is not None and low_freq_factor is not None and high_freq_factor is not None and factor is not None:
        low_freq_wavelen = original_context_length / low_freq_factor
        high_freq_wavelen = original_context_length / high_freq_factor
        wavelen = 2 * torch.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (original_context_length / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama
    positions = torch.arange(context_length)
    angles = positions[:, None] * inv_freq[None, :]  # (context_length, head_dim/2)
    angles = torch.cat([angles, angles], dim=1)  # (context_length, head_dim)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def compute_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings (RoPE) to the last dimension of x.
    Expects x of shape (B, num_heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)

class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(
            context_length: int,
            head_dim: int,
            rope_base: int,
            original_context_length: int = None,
            low_freq_factor: float = None,
            high_freq_factor: float = None,
            factor: float = None,
            dtype: torch.dtype = torch.float32
    ) -> tuple:
        key = (context_length, head_dim, rope_base, original_context_length,
               low_freq_factor, high_freq_factor, factor, dtype)
        if key not in SharedBuffers._buffers:
            # For full attention, we no longer require a causal mask.
            mask = torch.zeros(context_length, context_length)
            cos, sin = precompute_rope_params(
                head_dim, rope_base, context_length,
                original_context_length, low_freq_factor, high_freq_factor, factor
            )
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)
        return SharedBuffers._buffers[key]

#########################################
# MultiHead Attention Module (Full/Bidirectional)
#########################################

class MultiHeadAttention(nn.Module):
    """
    A standard multi-head attention module that computes queries, keys, and values via
    separate linear projections, applies rotary positional embeddings (RoPE) to queries and keys,
    and then computes scaled dot-product attention using full (bidirectional) attention.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        rope_base: int = 10_000,
        original_context_length: int = None,
        low_freq_factor: float = None,
        high_freq_factor: float = None,
        factor: float = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)

        mask, cos, sin = SharedBuffers.get_buffers(
            context_length, self.head_dim, rope_base,
            original_context_length, low_freq_factor, high_freq_factor, factor, dtype
        )
        # For full attention, the mask is a dummy (all zeros).
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_tokens, d_in)
        b, num_tokens, _ = x.shape
        q = self.W_q(x)  # (B, num_tokens, d_out)
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape to (B, num_heads, num_tokens, head_dim)
        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        q = compute_rope(q, self.cos, self.sin)
        k = compute_rope(k, self.cos, self.sin)

        # Compute scaled dot-product attention (full bidirectional)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # No causal mask is applied; all positions attend to all others.
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)  # (B, num_heads, num_tokens, head_dim)

        # Combine heads: reshape to (B, num_tokens, d_out)
        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        out = self.out_proj(context)
        return out

#########################################
# Transformer Block (using Full MultiHeadAttention)
#########################################

class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        context_length: int,
        n_heads: int,
        rope_base: int,
        original_context_length: int = None,
        low_freq_factor: float = None,
        high_freq_factor: float = None,
        factor: float = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            num_heads=n_heads,
            rope_base=rope_base,
            original_context_length=original_context_length,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            factor=factor,
            dtype=dtype
        )
        self.ff = FeedForward(emb_dim, emb_dim * 4, dtype)
        self.norm1 = nn.RMSNorm(emb_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(emb_dim, eps=1e-6)
        # self.norm3 = nn.RMSNorm(emb_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x

#########################################
# Encoder and Decoder (without kv_group)
#########################################

class Encoder(nn.Module):
    def __init__(self,
                 length: int,
                 n_layers: int,
                 chunk_size: int,
                 embed_dim: int,
                 # hidden_dim: int,
                 n_heads: int,
                 latent_dim: int,
                 conv = False,
                 flatten = True,
                 rope_base: int = 10_000,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.length = length
        self.chunk_size = chunk_size
        num_tokens = math.ceil(length / chunk_size)
        self.num_tokens = num_tokens
        self.n_heads = n_heads
        self.rope_base = rope_base
        # self.dtype = dtype
        self.flatten = flatten
        self.conv = conv
        self.latent_dim = latent_dim
        self.layernorm = nn.LayerNorm(embed_dim)
        # self.hidden_dim = hidden_dim

        self.embed = WeightEmbed(chunk_size, embed_dim, conv=conv, flatten=flatten)
        self.encoder_transformer = nn.Sequential(
            *[TransformerBlock(
                emb_dim=embed_dim,
                context_length=num_tokens,
                n_heads=n_heads,  # set number of heads as needed
                rope_base=rope_base,
                dtype=dtype
            ) for _ in range(n_layers)]
        )
        # self.fc_in = nn.Linear(chunk_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        # x = self.fc_in(x)
        # x = F.gelu(x)
        # print(x.shape)
        x = self.encoder_transformer(x)
        x = self.layernorm(x)
        return x

class Decoder(nn.Module):
    def __init__(self,
                 length: int,
                 n_layers: int,
                 chunk_size: int,
                 embed_dim: int,
                 n_heads: int,
                 latent_dim: int,
                 conv=False,
                 flatten=True,
                 rope_base: int = 10000,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.length = length
        self.chunk_size = chunk_size
        num_tokens = math.ceil(length / chunk_size)
        self.num_tokens = num_tokens
        self.original_length = length
        self.n_heads = n_heads
        self.rope_base = rope_base
        self.flatten = flatten
        self.conv = conv
        self.latent_dim = latent_dim

        self.decoder_transformer = nn.Sequential(
            *[TransformerBlock(
                emb_dim=embed_dim,
                context_length=num_tokens,
                n_heads=n_heads,  # set number of heads as needed
                rope_base=rope_base,
                dtype=dtype
            ) for _ in range(n_layers)]
        )
        # self.token_projector = nn.Linear(embed_dim, embed_dim)
        self.decode = WeightDecode(chunk_size, embed_dim, conv=conv)
        self.layernorm = nn.LayerNorm(embed_dim)
        # self.fc_out = nn.Linear(embed_dim, chunk_size)

    def forward(self, z: torch.Tensor, original_length: int = None) -> torch.Tensor:
        # z = self.layernorm(z)
        x = self.decoder_transformer(z)
        x = self.layernorm(x)
        # print(x.shape)
        # x = self.fc_out(x)
        # x =F.gelu(x)
        # x = self.token_projector(x)
        if original_length is None:
            original_length = self.original_length
        x = self.decode(x, original_length=original_length)
        return x
#
# #########################################
# # Integrated Weight VAE (using full attention and without kv_group)
# #########################################
#
# class WeightVAE(nn.Module):
#     """
#     A VAE for flattened weight vectors that tokenizes the input,
#     encodes via transformer blocks using full (bidirectional) attention,
#     and decodes back into a weight vector.
#     """
#     def __init__(self,
#                  n_layers: int,
#                  chunk_size: int,
#                  embed_dim: int,
#                  latent_dim: int,
#                  hidden_dim: int,
#                  length: int,
#                  dtype: torch.dtype = torch.float32,
#                  conv: bool = True):
#         super().__init__()
#         self.length = length
#         self.embed = WeightEmbed(chunk_size, embed_dim, conv=conv, flatten=True)
#         num_tokens = math.ceil(length / chunk_size)
#
#         # Encoder transformer: uses full attention
#         self.encoder_transformer = nn.Sequential(
#             *[TransformerBlock(
#                 emb_dim=embed_dim,
#                 context_length=num_tokens,
#                 n_heads=8,  # adjust as needed
#                 rope_base=10_000,
#                 dtype=dtype
#             ) for _ in range(n_layers)]
#         )
#         self.fc_mu = nn.Linear(embed_dim, latent_dim)
#         self.fc_logvar = nn.Linear(embed_dim, latent_dim)
#         # Decoder: project latent vector to initial tokens for transformer decoding.
#         self.latent_to_tokens = nn.Linear(latent_dim, hidden_dim)
#         self.decoder_transformer = nn.Sequential(
#             *[TransformerBlock(
#                 emb_dim=hidden_dim,
#                 context_length=num_tokens,
#                 n_heads=8,  # adjust as needed
#                 rope_base=10_000,
#                 dtype=dtype
#             ) for _ in range(n_layers)]
#         )
#         self.token_projector = nn.Linear(hidden_dim, embed_dim)
#         self.decode = WeightDecode(chunk_size, embed_dim, conv=conv)
#
#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         # x: (B, L) flattened weight vector
#         B, L = x.shape
#         tokens = self.embed(x)                # (B, num_tokens, embed_dim)
#         encoded = self.encoder_transformer(tokens)  # (B, num_tokens, embed_dim)
#         mu = self.fc_mu(encoded)
#         logvar = self.fc_logvar(encoded)
#         z = self.reparameterize(mu, logvar)
#         token_init = self.latent_to_tokens(z)
#         decoded_tokens = self.decoder_transformer(token_init)
#         decoded_tokens = self.token_projector(decoded_tokens)
#         recon_x = self.decode(decoded_tokens, original_length=L)
#         return recon_x, mu, logvar
