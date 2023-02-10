import torch
import torch.nn as nn
from einops.layers.torch import Rearrange



class ConvolutionalTokenEmbedding(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, kernel, stride, padding, size_div):
        super().__init__()
        self.size = image_size//size_div

        #2D convolution operation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        # flattened into size HiWi × Ci
        self.rearrange = Rearrange('b c h w -> b (h w) c', h=self.size, w=self.size)
        #normalized for input into the subsequent Transformer blocks of stage i
        self.layer_norm = nn.LayerNorm(out_channels)    #
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.rearrange(x)
        x = self.layer_norm(x)
        return x
