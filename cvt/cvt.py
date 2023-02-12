import torch
import torch.nn as nn
from einops import repeat

from cvt.conv_token_embedding import ConvolutionalTokenEmbedding
from cvt.conv_transformer_blocks import ConvolutionalTransformerBlocks


class CvT(nn.Module):
    def __init__(self,
                 img_size,  # Integer, height and width of the image.
                 # Integer, number of channels of the input image.
                 in_channels,
                 # Integer, number of classes to classify the images into.
                 num_classes,
                 # Integer, dimensionality of the hidden state for the self-attention mechanism in the transformer block.
                 dim=64,
                 # List of Integers, kernel size of the Convolutional Embedding layer at each stage
                 kernels=[7, 3, 3],
                 # List of Integers, stride size of the Convolutional Embedding layer at each stage
                 strides=[4, 2, 2],
                 # List of Integers, number of heads in the Multi-Head Attention for each stage
                 heads=[1, 3, 6],
                 # List of Integers, number of Transformer blocks in each stage of the model
                 depth=[1, 2, 10],
                 pool='cls',  # String, pooling method to use after the final stage of the model
                 # The possible values are
                 # 'cls'  (pooling over the first token of the sequence)
                 # 'mean' (mean pooling over the entire sequence)
                 dropout=0.,  # Float, dropout rate to use after each transformer block
                 scale_MLP=4  # Integer, scaling factor for the dimensionality of the MLP in the Self-Attention Mechanism
                 ):
        super().__init__()
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim

        ##### Stage 1 #######
        self.stage1_conv_embed = ConvolutionalTokenEmbedding(img_size=img_size,
                                                             in_channels=in_channels,
                                                             out_channels=dim,
                                                             kernel=kernels[0],
                                                             stride=strides[0],
                                                             padding=2,
                                                             size_div=4)
        self.stage1_conv_transformer = ConvolutionalTransformerBlocks(dim=dim,
                                                                      img_size=img_size//4,
                                                                      depth=depth[0],
                                                                      heads=heads[0],
                                                                      dim_head=self.dim,
                                                                      mlp_dim=dim * scale_MLP,
                                                                      dropout=dropout)

        ##### Stage 2 #######
        in_channels = dim
        scale = heads[1]//heads[0]
        dim = scale*dim
        self.stage2_conv_embed = ConvolutionalTokenEmbedding(img_size=img_size,
                                                             in_channels=in_channels,
                                                             out_channels=dim,
                                                             kernel=kernels[1],
                                                             stride=strides[1],
                                                             padding=1,
                                                             size_div=8)
        self.stage2_conv_transformer = ConvolutionalTransformerBlocks(dim=dim,
                                                                      img_size=img_size//8,
                                                                      depth=depth[1],
                                                                      heads=heads[1],
                                                                      dim_head=self.dim,
                                                                      mlp_dim=dim * scale_MLP,
                                                                      dropout=dropout)

        ##### Stage 3 #######
        in_channels = dim
        scale = heads[2] // heads[1]
        dim = scale * dim
        self.stage3_conv_embed = ConvolutionalTokenEmbedding(img_size=img_size,
                                                             in_channels=in_channels,
                                                             out_channels=dim,
                                                             kernel=kernels[2],
                                                             stride=strides[2],
                                                             padding=1,
                                                             size_div=16)
        self.stage3_conv_transformer = ConvolutionalTransformerBlocks(dim=dim,
                                                                      img_size=img_size//16,
                                                                      depth=depth[2],
                                                                      heads=heads[2],
                                                                      dim_head=self.dim,
                                                                      mlp_dim=dim * scale_MLP,
                                                                      dropout=dropout,
                                                                      last_stage=True)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, img):

        xs = self.stage1_conv_embed(img)
        xs = self.stage1_conv_transformer(xs)

        xs = self.stage2_conv_embed(xs)
        xs = self.stage2_conv_transformer(xs)

        xs = self.stage3_conv_embed(xs)
        b, n, _ = xs.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        xs = torch.cat((cls_tokens, xs), dim=1)
        xs = self.stage3_conv_transformer(xs)
        xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]

        xs = self.mlp_head(xs)
        return xs
