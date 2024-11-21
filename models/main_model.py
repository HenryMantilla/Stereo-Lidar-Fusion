import torch.nn as nn
import torch
import timm

from models.modules import ChannelAttention
from models.decoders import ConvDecoderWithConvexUp
from models.encoders.PVT import PvtFamily

class DepthFusion(nn.Module):
    def __init__(self, in_chans_transformer, convnext_pretrained):
        super().__init__()

        self.num_stages = 3
        self.convnext_encoder = timm.create_model('convnextv2_atto.fcmae', convnext_pretrained)

        for i in range(4):
            setattr(self, f'channel_attn_{i}', ChannelAttention(in_chans=in_chans_transformer[i], ratio=8))

        self.pvt_family = PvtFamily()
        self.transformer_backbone = self.pvt_family.initialize_model(config_name="pvt_medium", patch_size=4, in_chans=40, num_stages=self.num_stages)

        # [96,192,384,768] -> (modify to factor of 4 like 32,64) + [64, 128, 320, 512]
        self.decoder = ConvDecoderWithConvexUp(in_chans=[320, 576, 288, 144, 72]) #[1280, 896,448, 224]

        self.out_conv = nn.Sequential(
            nn.Conv2d(160,1,1),
            nn.Sigmoid()
        )

        self.expand_ch = nn.Conv2d(1,3, kernel_size=1)

        self.stereo_CA = []
        self.sparse_CA = []
        
    #Input both as 3 channel 
    def forward(self, stereo, sparse):
        out_stereo, stereo_intermediates = self.convnext_encoder.forward_intermediates(self.expand_ch(stereo)) # [96,192,384,768]
        out_sparse, sparse_intermediates = self.convnext_encoder.forward_intermediates(self.expand_ch(sparse))

        #if 240x1216 output HxW => [(60,384), (30,152), (15,76), (7,38)] H/4 x W/4
        cross_dom_attn = self.transformer_backbone(stereo_intermediates[1],sparse_intermediates[1]) # H/32 x W/32 (240,1216)

        for i in range(self.num_stages+1):
            CA = getattr(self, f'channel_attn_{i}')
            self.stereo_CA.append(CA(stereo_intermediates[i+1])*stereo_intermediates[i+1])
            self.sparse_CA.append(CA(sparse_intermediates[i+1])*sparse_intermediates[i+1])

        decoder_out = self.decoder(self.sparse_CA, cross_dom_attn)

        return decoder_out
        

batch_size = 2
height = 256
width = 1216

stereo = torch.rand(batch_size, 1, height, width)  # Stereo input (3 channels)
sparse = torch.rand(batch_size, 1, height, width)  # Sparse input (3 channels)
in_chans = [40,80,160,320] #[96,192,384,768] 

model = DepthFusion(in_chans_transformer=in_chans, convnext_pretrained=True)

output = model(stereo, sparse)
print('We did it',output.shape)