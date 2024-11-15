import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple


def tconv_block(in_ch, out_ch, kernel, stride=1, padding=0):

    tconv_blk = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding),
                              nn.BatchNorm2d(out_ch),
                              nn.ReLU(inplace=True))
    
    return tconv_blk

def conv_block(in_ch,out_ch, kernel, stride=1, padding=0):

    conv_blk = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
                              nn.BatchNorm2d(out_ch),
                              nn.ReLU(inplace=True))
    
    return conv_blk


class ConvexUpsampling(nn.Module):
    def __init__(self, in_chans, upsample_factor=2):
        super().__init__()

        self.upsampler_conv = nn.Sequential(nn.Conv2d(in_chans, 256, 3, 1, 1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))

    def forward(self, x):

        B,C,H,W = x.shape

        mask = self.upsampler_conv(x)
        mask = mask.view(B, 1, 9, self.upsample_factor, self.upsample_factor, H, W)
        mask = torch.softmax(mask, dim=2)

        upsample_x = F.unfold(self.upsample_factor * x, [3, 3], padding=1)
        upsample_x = upsample_x.view(B, C, 9, 1, 1, H, W)

        upsample_x = torch.sum(mask * upsample_x, dim=2).permute(0, 1, 4, 2, 5, 3)

        return upsample_x.reshape(B, C, self.upsample_factor * H, self.upsample_factor * W)


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=16):
        super().__init__()

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        self.net = nn.Sequential(nn.Conv2d(in_ch, in_ch//ratio, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(in_ch//ratio, in_ch, 1))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_pool = self.net(self.avg_pooling(x))
        max_pool = self.net(self.max_pooling(x))

        ch_attn = max_pool + avg_pool

        return self.sigmoid(ch_attn)


class PatchEmbedding(nn.Module):  
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialReductionAttention(nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
            super().__init__()
            assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

            self.dim = dim
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5

            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

            self.sr_ratio = sr_ratio
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)

        def forward(self, x_stereo, x_sparse, H, W):
            B, N, C = x_sparse.shape
            q = self.q(x_sparse).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            if self.sr_ratio > 1:
                x_ = x_stereo.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x_stereo).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x