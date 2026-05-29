import torch
import torch.nn as nn
import torch.nn.functional as F



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



class TubeletEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 tubelet_size=2,
                 conv=True,
                 flatten=True):
        super().__init__()
        self.conv = conv
        if conv:
            self.proj = nn.Conv3d(
                in_channels=in_chans,
                out_channels=embed_dim,
                kernel_size=(tubelet_size, patch_size, patch_size),
                stride=(tubelet_size, patch_size, patch_size))
        else:
            self.proj = nn.Linear(in_chans * patch_size**2, embed_dim)
        self.flatten = flatten
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

    def forward(self, x):
        if self.conv:
            x = self.proj(x)
        else:
            tb, p = self.tubelet_size, self.patch_size
            assert x.shape[3] == x.shape[4] and x.shape[3] % p == 0
            assert x.shape[2] % tb == 0
            ts, hs, ws = x.shape[2] // tb, x.shape[3] // p, x.shape[4] // p
            x = x.reshape(shape=(x.shape[0], x.shape[1], ts, tb, hs, p, ws, p))
            x = torch.einsum('nctbhpwq->nthwcpq', x)
            x = x.reshape(shape=(x.shape[0], ts * hs * ws, tb * p**2 * x.shape[1]))
            x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x

class TubeletDecode(nn.Module):
    def __init__(
        self,
        patch_size=16,
        tublet_size=8,
        out_channels=3,
        in_embed_dim=768,
        conv=True,
    ):
        super().__init__()
        self.tublet_size = tublet_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.conv = conv
        self.in_embed_dim = in_embed_dim
        if self.conv:
            self.proj = nn.ConvTranspose3d(
                in_channels=in_embed_dim,
                out_channels=out_channels,
                kernel_size=(tublet_size, patch_size, patch_size),
                stride=(tublet_size, patch_size, patch_size),
            )
        else:
            self.proj = nn.Linear(
                in_features=in_embed_dim,
                out_features=patch_size * patch_size * tublet_size * out_channels,
            )

    def forward(self, x, grid_size):
        c = self.out_channels
        tb, p, p = self.tublet_size, self.patch_size, self.patch_size
        t, h, w = grid_size
        if self.conv:
            # print(x.shape)
            x = x.reshape(shape=(x.shape[0], t, h, w, self.in_embed_dim))
            x = x.permute(0, 4, 1, 2, 3) # b, c, t, h, w
            return self.proj(x)
        else:
            x = self.proj(x)
            x = x.reshape(shape=(x.shape[0], t, h, w, tb, p, p, c))
            x = torch.einsum("bthwpqrc->bctphqwr", x)
            return  x.reshape(shape=(x.shape[0], c, t * tb, h * p, w * p))