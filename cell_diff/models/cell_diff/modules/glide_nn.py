import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import unpatchify

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels, 
        patch_size, 
        num_heads=1, 
        text_emd_dims=480, 
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.norm_text = nn.LayerNorm(text_emd_dims, elementwise_affine=False, eps=1e-6)
        
        self.qkv = nn.Linear(channels, channels * 3)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.text_kv = nn.Linear(text_emd_dims, channels * 2)
        
        self.proj_out = nn.Linear(channels, channels * patch_size * patch_size)
        
        self.img_patch_emd = nn.Conv2d(channels, channels, kernel_size=patch_size, stride=patch_size, bias=True)
        self.patch_size = patch_size

    def forward(self, x_in, text_emd):
        x = self.img_patch_emd(x_in)
        
        B, C, *spatial = x.shape
        
        x = x.view(B, C, -1).transpose(1, 2)
        B, N, C = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        B_t, N_t, C_t = text_emd.shape
        text_kv = self.text_kv(self.norm_text(text_emd)).reshape(B_t, N_t, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        text_k, text_v = text_kv.unbind(0)
        
        k = torch.cat([k, text_k], dim=-2)
        v = torch.cat([v, text_v], dim=-2)

        h = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        h = h.transpose(1, 2).reshape(B, N, C)
        h = self.proj_out(h)
        h = self.proj_drop(h)
        
        h = unpatchify(h, self.channels, self.patch_size)
        x_in = x_in + h

        return x_in