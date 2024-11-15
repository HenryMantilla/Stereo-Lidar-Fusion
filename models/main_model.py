import torch.nn as nn
import timm

from models.modules import ChannelAttention
from models.decoders import ConvDecoderWithConvexUp
from models.encoders.PVT import PvtFamily

class DepthFusion(nn.Module):
    def __init__(self, in_chans, convnext_pretrained, features_only):
        super().__init__()

        self.convnext_encoder = timm.create_model('convnextv2_tiny.fcmae', convnext_pretrained, features_only)

        for i in range(4):
            # in_chans of convnext [96,192,384,768] 
            setattr(self, f'channel_attn_{i}', ChannelAttention(in_chans=in_chans[i], ratio=8))
        
        #use first output of convnextv2 which is H/4 x W/4 of image_size
        self.transformer_backbone = PvtFamily.initialize_model("pvt_medium", patch_size=4, in_chans=96, num_stages=3)

        # [96,192,384,768] -> (modify to factor of 4 like 32,64) + [64, 128, 320, 512]
        self.decoder = ConvDecoderWithConvexUp(in_chans=[1280, 896,448, 224])

        self.out_conv = nn.Sequential(
            nn.Conv2d(160,1,1),
            nn.Sigmoid()
        )

        self.expand_ch = nn.Conv2d(1,3, kernel_size=1)

        self.stereo_CA = []
        self.sparse_CA = []
        
    #Input both as 3 channel 
    def forward(self, stereo, sparse):

        stereo_features = self.convnext_encoder(self.expand_ch(stereo)) # [96,192,384,768]
        sparse_features = self.convnext_encoder(self.expand_ch(sparse))

        #if 240x1216 output HxW => [(60,384), (30,152), (15,76), (7,38)] H/4 x W/4
        # inter domain attention
        interdomain_attn = self.transformer_backbone(stereo_features[0],sparse_features[0]) # H/32 x W/32 (240,1216)

        for i in range(self.num_stages):
            B,C,H,W = stereo_features[0].shape

            CA = getattr(self, f'channel_attention{i}')
            self.stereo_CA.append(CA(stereo_features[i])*stereo_features[i])
            self.sparse_CA.append(CA(sparse_features[i])*sparse_features[i])

        decoder_out = self.decoder(self.sparse_CA, interdomain_attn)

        return decoder_out

