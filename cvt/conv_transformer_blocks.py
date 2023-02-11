import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from cvt.conv_projection import ConvolutionalProjection
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP(nn.Module):
    """
    MLP (i.e. fully connected) Head is utilized upon
    the classification token of the final stage output 
    to predict the class.
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionalTransformerBlocks(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.img_size = img_size
        self.last_stage = last_stage
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvolutionalProjection(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=self.last_stage)),
                PreNorm(dim, MLP(dim, mlp_dim, dropout=dropout))
            ]))
        self.rearrange = Rearrange('b (h w) c -> b c h w', h = self.img_size, w = self.img_size)
        

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        if not self.last_stage:
            x = self.rearrange(x)
        return x