from typing import Optional
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .glide_nn import AttentionBlock


def exists(val):
    return val is not None

def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"

def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d

class Upsample(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        factor: int = 2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = default(dim_out, dim)
        assert isinstance(dim_out, int) and dim_out > 0, 'dim_out must be a positive integer'
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(
    dim,
    dim_out = None,
    factor = 2
):
    dim_out = default(dim_out, dim)
    assert isinstance(dim_out, int) and dim_out > 0, 'dim_out must be a positive integer'
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = factor, p2 = factor),
        nn.Conv2d(dim * (factor ** 2), dim_out, 1)
    )

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()        
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            assert isinstance(scale_shift, tuple) and len(scale_shift) == 2, 'scale and shift must be a tuple of len 2'
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

class ResAttBlock(nn.Module): 
    def __init__(self, 
                 dim, 
                 dim_out, 
                 patch_size, 
                 emb_dim=None, 
                 num_heads=1, 
                 attn_drop=0.0): 
        super().__init__()
        self.res_block = ResnetBlock(dim, dim_out, emb_dim)
        self.att_block = AttentionBlock(dim_out, patch_size, num_heads, emb_dim, attn_drop)
        
    def forward(self, x, time_emb, text_emb):
        x = self.res_block(x, time_emb)
        x = self.att_block(x, text_emb)

        return x