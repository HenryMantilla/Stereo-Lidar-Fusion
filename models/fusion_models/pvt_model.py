import timm
import torch
import torch.nn as nn

from models.encoders.PVT import PvtFamily
from models.modules import CBAM, ConvBlock
from models.dyspn import Dynamic_deformablev2

class DepthFusionPVT(nn.Module):
    def __init__(self, convnext_pretrained):
        super().__init__()

        self.expand_ch = nn.Conv2d(4, 3, kernel_size=1)
        self.convnext_encoder = timm.create_model('convnextv2_nano.fcmae', convnext_pretrained)
        #self.conv1x1 = nn.Conv2d(640, 1024, kernel_size=1)
        #self.decoder_1 = nn.PixelShuffle(32)

        pvt_family = PvtFamily()
        self.pvt = pvt_family.initialize_model(config_name="pvt_medium", patch_size=4, in_chans=4, num_stages=4)
        self.decoder = ConvDecoder(input_channels=512, output_channels=48)

        self.refinement = Dynamic_deformablev2(iteration=6)

    def forward(self, rgb, stereo, lidar):

        input_cnn = self.expand_ch(torch.cat([rgb, stereo], dim=1))
        _, intermediates_cnn = self.convnext_encoder.forward_intermediates(input_cnn)
        #out_cnn = self.conv1x1(out_cnn)
        #cnn_depth = self.decoder_1(out_cnn)
        
        rgb_lidar = torch.cat([rgb, lidar], dim=1)

        out_pvt, intermediate_pvt = self.pvt(rgb_lidar, intermediates_cnn[1:])
        init_depth, confidence, guide = self.decoder(out_pvt, intermediate_pvt, rgb_lidar)

        pred = self.refinement(init_depth,
                               guide[:, 0:24, :, :],
                               guide[:, 24:48, :, :],
                               confidence,
                               lidar)
        
        return init_depth, pred, confidence
    
"""
class ConvDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvDecoder, self).__init__()

        #Up+Conv+BN+ReLU+CBAM block
        self.shuffle = nn.PixelShuffle(16)
        def upsample_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
                CBAM(out_ch)  
            )

        self.stage1 = upsample_conv(input_channels+640, 256)
        #self.conv_confidence = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.init_depth = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1, groups=1),  # Depthwise
                nn.Conv2d(64, 32, kernel_size=1),  
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

        self.last_conv = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm2d(16),
                                       nn.Conv2d(16, output_channels, kernel_size=3, padding=1))
        
        self.conv_confidence = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1),
                                       nn.Sigmoid())

    def forward(self, x, feats_cnn, feats_vit):

        x = torch.cat([x, feats_cnn[3]], dim=1)
        x = self.stage1(x)

        x = self.shuffle(x)

        intermediate_upsamples = [x,x,x]
        confidence_map = self.conv_confidence(x)
        init_depth = self.init_depth(x)
        x = self.last_conv(x)

        return init_depth, confidence_map, x, intermediate_upsamples
"""

class ConvDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvDecoder, self).__init__()

        #Up+Conv+BN+ReLU+CBAM block
        def upsample_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
                CBAM(out_ch)  
            )

        self.stage1 = upsample_conv(input_channels, 512)   
        self.stage2 = upsample_conv(512+320, 256)               
        self.stage3 = upsample_conv(256+128, 128)             
        self.stage4 = upsample_conv(128+64, 64)  

        self.last_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.init_depth = ConvBlock(64, 32)
        self.init_depth_f = ConvBlock(32+4, 1)

        self.conv_confidence = ConvBlock(64, 32)
        self.conv_confidence_f = ConvBlock(32+4, 1)

        self.conv_guide = ConvBlock(64, 32)
        self.conv_guide_f = ConvBlock(32+4, output_channels)

    def forward(self, x, feats, input_vit):

        #x = torch.cat([x, feats[3]], dim=1)
        x = self.stage1(x)

        x = torch.cat([x, feats[2]], dim=1)
        x = self.stage2(x)

        x = torch.cat([x, feats[1]], dim=1)
        x = self.stage3(x)

        x = torch.cat([x, feats[0]], dim=1)
        x = self.stage4(x)
        
        x = self.last_upsample(x)

        init_depth = self.init_depth(x)
        init_depth = self.init_depth_f(torch.cat([init_depth, input_vit], dim=1))

        confidence_map = self.conv_confidence(x)
        confidence_map = self.conv_confidence_f(torch.cat([confidence_map, input_vit], dim=1))

        guide =  self.conv_guide(x)
        guide = self.conv_guide_f(torch.cat([guide, input_vit], dim=1))

        return init_depth, confidence_map, guide

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNetCBAM(nn.Module):
    def __init__(self, input_channels=4, out_channels=1, base_ch=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        #Encoder
        self.encoders = nn.ModuleList()
        self.cbams_enc = nn.ModuleList()
        channels = [input_channels] + [base_ch * (2 ** i) for i in range(4)]
        for i in range(4):
            self.encoders.append(DoubleConv(channels[i], channels[i + 1]))
            self.cbams_enc.append(CBAM(channels[i + 1]))

        #Bottleneck
        self.bottleneck = DoubleConv(channels[-1], base_ch * 16)
        self.cbam_bottleneck = CBAM(base_ch * 16)

        #Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.cbams_dec = nn.ModuleList()
        for i in range(3, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(base_ch * (2 ** (i + 1)), base_ch * (2 ** i), kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(base_ch * (2 ** (i + 1)), base_ch * (2 ** i)))
            self.cbams_dec.append(CBAM(base_ch * (2 ** i)))

        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        enc_features = []

        for enc, cbam in zip(self.encoders, self.cbams_enc):
            x = enc(x)
            x = cbam(x)
            enc_features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.cbam_bottleneck(x)

        for i in range(len(self.decoders)):
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_features[-(i + 1)]], dim=1)
            x = self.decoders[i](x)
            x = self.cbams_dec[i](x)

        return self.out_conv(x)