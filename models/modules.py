import torch
import torch.nn as nn
import torch.nn.functional as F

def tconv_block(in_ch, out_ch, kernel, stride=1, padding=0):

    tconv_blk = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding),
                              nn.BatchNorm2d(out_ch),
                              nn.ReLU(inplace=True))
    
    return tconv_blk

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, normalization=True, act=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=not normalization)]
        if normalization:
            layers.append(nn.BatchNorm2d(out_ch))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_chans, ratio=16):
        super().__init__()

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        self.net = nn.Sequential(nn.Conv2d(in_chans, in_chans//ratio, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(in_chans//ratio, in_chans, 1))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_pool = self.net(self.avg_pooling(x))
        max_pool = self.net(self.max_pooling(x))

        ch_attn = max_pool + avg_pool

        return self.sigmoid(ch_attn) 


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.sigmoid(self.conv(x))

        return x


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction=16, kernel_size=7):
        super().__init__()
        
        self.conv_blk = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels)

        )

        self.channel_attention = ChannelAttention(input_channels, ratio=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        self.act = nn.ReLU()

    def forward(self, x):

        residual = x

        x = self.conv_blk(x)
        ca = self.channel_attention(x)
        x = x * ca

        sa = self.spatial_attention(x)
        x = x * sa

        x += residual

        return self.act(x)