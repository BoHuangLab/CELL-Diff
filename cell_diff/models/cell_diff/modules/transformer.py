import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class TransformerBlock(nn.Module):
    """
    A transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, 
                 hidden_size, 
                 num_heads, 
                 mlp_ratio=4.0, 
                 attn_drop=0.0, 
                 mlp_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=mlp_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask=None):
        
        # Apply shift when mask is True
        
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device).bool()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        shift_msa = shift_msa.unsqueeze(1).repeat(1, x.shape[1], 1)
        shift_msa[~mask] = 0

        scale_msa = scale_msa.unsqueeze(1).repeat(1, x.shape[1], 1)
        scale_msa[~mask] = 0

        gate_msa = gate_msa.unsqueeze(1).repeat(1, x.shape[1], 1)
        gate_msa[~mask] = 0

        shift_mlp = shift_mlp.unsqueeze(1).repeat(1, x.shape[1], 1)
        shift_mlp[~mask] = 0

        scale_mlp = scale_mlp.unsqueeze(1).repeat(1, x.shape[1], 1)
        scale_mlp[~mask] = 0

        gate_mlp = gate_mlp.unsqueeze(1).repeat(1, x.shape[1], 1)
        gate_mlp[~mask] = 0

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


def unpatchify(x, out_channels, patch_size):
    """
    x: (N, T, patch_size**2 * C)
    imgs: (N, C, H, W)
    """
    c = out_channels
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        shift = shift.unsqueeze(1).repeat(1, x.shape[1], 1)
        scale = scale.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x