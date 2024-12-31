import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.frame_utils import resize_image
from utils.model_utils import download_sam_checkpoint

from models.modules import CBAM, ConvBlock
from models.encoders.SwinV2 import SwinTransformerV2
from models.dyspn import Dynamic_deformablev2
from models.segment_anything import sam_model_registry

class DepthFusionSAM(nn.Module):
    def __init__(self, convnext_pretrained):
        super().__init__()

        self.expand_ch = nn.Conv2d(4, 3, kernel_size=1)
        self.convnext_encoder = timm.create_model('convnextv2_nano.fcmae', convnext_pretrained)

        sam = sam_model_registry['vit_b'](checkpoint='sam_checkpoints/sam_vit_b_01ec64.pth')
        self.sam_encoder = sam.image_encoder()
        self.resize_sam_fmap = ResizeFeatureMap()
        self.fuse_features = FuseFeatures(256, 80)

        self.swinv2 = SwinTransformerV2(img_size=(256,1280), patch_size=4, in_chans=4, embed_dim=96, window_size=8)
        self.decoder = ConvDecoder(input_channels=768, output_channels=48)

        self.refinement = Dynamic_deformablev2(iteration=6)

    
    def forward(self, rgb, stereo, lidar):
        
        rgb_sam = resize_image(rgb)
        sam_enc_embedding = self.sam_encoder(rgb_sam) # 256x64x64
        sam_enc_embedding = self.resize_sam_fmap(sam_enc_embedding, size=(64,304))

        input_cnn = self.expand_ch(torch.cat([rgb, stereo], dim=1))
        _, intermediates_cnn = self.convnext_encoder.forward_intermediates(input_cnn)
        intermediates_cnn = intermediates_cnn[1:]

        fused_features = self.fuse_features([intermediates_cnn[0], sam_enc_embedding])
        intermediates_cnn[0] = fused_features

        rgb_stereolidar = torch.cat([rgb, stereo], dim=1)
        out_swin, intermediates_swin = self.swinv2(rgb_stereolidar, intermediates_cnn)
        init_depth, confidence, guide = self.decoder(out_swin, intermediates_swin, rgb_stereolidar)

        pred = self.refinement(init_depth,
                        guide[:, 0:24, :, :],
                        guide[:, 24:48, :, :],
                        confidence,
                        lidar)
        
        return init_depth, pred, confidence
    

class ConvDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvDecoder, self).__init__()

        #Up+Conv+BN+GELU+CBAM block
        def upsample_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
                CBAM(out_ch)  
            )

        self.stage1 = upsample_conv(input_channels, 512)   
        self.stage2 = upsample_conv(512+384, 256)               
        self.stage3 = upsample_conv(256+192, 128)             
        self.stage4 = upsample_conv(128+96, 64)  

        self.last_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.init_depth = ConvBlock(64, 32)
        self.init_depth_f = ConvBlock(32+4, 1)

        self.conv_confidence = ConvBlock(64, 32)
        self.conv_confidence_f = ConvBlock(32+4, 1)

        self.conv_guide = ConvBlock(64, 32)
        self.conv_guide_f = ConvBlock(32+4, output_channels)

    def forward(self, x, feats, input_vit):

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


class ResizeFeatureMap(nn.Module):
    def __init__(self):
        super(ResizeFeatureMap, self).__init__()

    def forward(self, x, size=(64, 304)):
        return F.interpolate(x, size, mode='bilinear', align_corners=False)


class FuseFeatures(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FuseFeatures, self).__init__()

        self.cbam = CBAM(input_channels)
        self.reduce_ch = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, *features):

        x = torch.cat(features, dim=1)
        x = self.cbam(x)
        x = self.reduce_ch(x)

        return x
    