import torch
import torch.nn as nn
import torch.nn.functional as F
from models.igev_plus_plus.core.igev_stereo import IGEVStereo
#from models.igev.core.igev_stereo import IGEVStereo
from utils.frame_utils import disparity_to_depth
from models.encoders.PVT import PvtFamily
from models.modules import CBAM, ConvBlock
from models.dyspn import Dynamic_deformablev2

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict

class DepthFusionPVT(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.stereo_matching = IGEVStereo(args)

        checkpoint = torch.load('kitti2015_igevplusplus.pth', map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = remove_module_prefix(state_dict)
        
        #Load the state dict into the stereo_matching submodule
        self.stereo_matching.load_state_dict(state_dict)

        self.rgb_conv = ConvBlock(in_ch=3, out_ch=48, normalization=False)
        self.depth_conv = ConvBlock(in_ch=1, out_ch=16, normalization=False)
        self.disp_conv = ConvBlock(in_ch=1, out_ch=16, normalization=False)

        self.rgb_disp_conv = ConvBlock(in_ch=64, out_ch=64, normalization=False)
        self.rgb_depth_conv = ConvBlock(in_ch=64, out_ch=64, normalization=False)

        self.resnet_blks_disp = nn.Sequential(ResidualBlock(in_ch=64, out_ch=64, downsample=False),
                                              ResidualBlock(in_ch=64, out_ch=64, downsample=False),
                                              ResidualBlock(in_ch=64, out_ch=64, downsample=False))

        self.resnet_blks_disp2 = nn.Sequential(ResidualBlock(in_ch=64, out_ch=128, downsample=True),
                                              ResidualBlock(in_ch=128, out_ch=128, downsample=False),
                                              ResidualBlock(in_ch=128, out_ch=128, downsample=False),
                                              ResidualBlock(in_ch=128, out_ch=128, downsample=False))
        
        self.resnet_blks_depth = nn.Sequential(ResidualBlock(in_ch=64, out_ch=64, downsample=False),
                                              ResidualBlock(in_ch=64, out_ch=64, downsample=False),
                                              ResidualBlock(in_ch=64, out_ch=64, downsample=False))

        self.resnet_blks_depth2 = nn.Sequential(ResidualBlock(in_ch=64, out_ch=128, downsample=True),
                                                ResidualBlock(in_ch=128, out_ch=128, downsample=False),
                                                ResidualBlock(in_ch=128, out_ch=128, downsample=False),
                                                ResidualBlock(in_ch=128, out_ch=128, downsample=False))
        # PVT Backbone
        pvt_family = PvtFamily()
        self.pvt = pvt_family.initialize_model(config_name="pvt_small", patch_size=2, in_chans=128, num_stages=4)

        self.conv_decoder = ConvDecoder(in_channels=512, out_channels=48)

        #self.fusion_depth = ConvBlock(2,1, normalization=False, act=True)
        self.dyspn = Dynamic_deformablev2(iteration=6)

    def forward(self, rgb_left_aug, rgb_left, rgb_right, lidar, width):

        disparity = self.stereo_matching(rgb_left, rgb_right, iters=16, test_mode=True)
        disparity = torch.clamp(disparity, min=2.0, max=192.0)

        #disp2depth = disparity_to_depth(disparity, width) #the best rmse is with this commented and use disparity domain

        enc_rgb = self.rgb_conv(rgb_left_aug)
        enc_disp = self.disp_conv(disparity)
        enc_depth = self.depth_conv(lidar)

        enc_rgb_disp = self.rgb_disp_conv(torch.cat([enc_rgb, enc_disp], dim=1))
        enc_rgb_depth = self.rgb_depth_conv(torch.cat([enc_rgb, enc_depth], dim=1))

        disp_feats = self.resnet_blks_disp(enc_rgb_disp)
        depth_feats = self.resnet_blks_depth(enc_rgb_depth)

        disp_feats2 = self.resnet_blks_disp2(disp_feats)
        depth_feats2 = self.resnet_blks_depth2(depth_feats)

        mid_features = self.pvt(disp_feats2, depth_feats2)

        residual_depth, residual_disp, confidence, guide = self.conv_decoder(mid_features, enc_rgb_disp, enc_rgb_depth, disp_feats, depth_feats, disp_feats2, depth_feats2)

        #enhanced_depth = disp2depth + residual_depth
        enhanced_depth = disparity_to_depth(disparity+residual_disp, width) + residual_depth

        final_pred = self.dyspn(enhanced_depth,
                                guide[:, 0:24, :, :],
                                guide[:, 24:48, :, :],
                                confidence,
                                lidar) 

        #return disp2depth, enhanced_depth, final_pred, residual_depth
        return disparity, enhanced_depth, final_pred, residual_depth, residual_disp

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        """
        Residual block with optional 1x1 convolution for channel matching.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        stride = 2 if downsample else 1
        
        # Main branch
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU(inplace=True)

        # Shortcut branch
        if downsample or in_ch != out_ch:
            self.shortcut = (
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
                if in_ch != out_ch else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        Forward pass for the residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        identity = self.shortcut(x)  # Adjust channels if necessary

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Add skip connection
        out = self.activation(out)
        
        return out


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        #Up+Conv+BN+ReLU+CBAM block
        def upsample_conv(in_ch, out_ch, use_cbam=True, reduction=16):
            layers = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch)
            ]
            
            if use_cbam:
                layers.append(CBAM(out_ch, reduction=reduction))
            
            return nn.Sequential(*layers)

        self.stage1 = upsample_conv(in_channels, 256, reduction=16)   
        self.stage2 = upsample_conv(256+320, 128, reduction=8)               
        self.stage3 = upsample_conv(128+128, 64, reduction=4)             
        self.stage4 = upsample_conv(64+64, 64, reduction=4)  
        self.stage5 = upsample_conv(64+256, 64, reduction=4)

        self.conv_res_disp1 = ConvBlock(64+64, 64)
        self.conv_res_disp_f1 = ConvBlock(64+64, 1, normalization=True, act=False)

        self.conv_res_disp2 = ConvBlock(64+64,64)
        self.conv_res_disp_f2 = ConvBlock(64+64, 1, normalization=True, act=False)

        self.conv_confidence = ConvBlock(64+128, 64)
        self.conv_confidence_f = ConvBlock(64+128, 1)

        self.conv_guide = ConvBlock(64+128, 64)
        self.conv_guide_f = ConvBlock(64+128, out_channels, normalization=False, act=False)
        
        self.act = nn.Sigmoid()

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f
    
    def forward(self, feats, enc_rgb_disp, enc_rgb_depth, disparity, depth, disparity_2, depth_2):

        x = self.stage1(feats[3]) # H/16 x W/16

        x = torch.cat([x, feats[2]], dim=1)
        x = self.stage2(x) # H/8 x W/8

        x = x = torch.cat([x, feats[1]], dim=1) 
        x = self.stage3(x) # H/4 x W/4

        x = x = torch.cat([x, feats[0]], dim=1) 
        x = self.stage4(x) # H/2 x W/2


        x = self._concat(torch.cat([disparity_2, depth_2], dim=1), x)
        x = self.stage5(x) # H x W

        #residual_depth = self.conv_res_disp1(self._concat(x, torch.cat([disparity, depth], dim=1)))
        residual_depth = self.conv_res_disp1(self._concat(x, depth))
        residual_depth = self.conv_res_disp_f1(self._concat(residual_depth, enc_rgb_depth))

        residual_disp = self.conv_res_disp2(self._concat(x, disparity))
        residual_disp = self.conv_res_disp_f2(self._concat(residual_disp, enc_rgb_disp))

        confidence = self.conv_confidence(self._concat(x, torch.cat([depth, disparity], dim=1)))
        confidence = self.conv_confidence_f(self._concat(confidence, torch.cat([enc_rgb_depth, enc_rgb_disp], dim=1)))

        guide = self.conv_guide(self._concat(x, torch.cat([depth, disparity], dim=1)))
        guide = self.conv_guide_f(self._concat(guide, torch.cat([enc_rgb_depth, enc_rgb_disp], dim=1)))

        #return 0.4*self.act(residual_disp)-0.2, 1.2*self.act(residual_depth)-0.6 
        res_disp = (self.act(residual_disp) - 0.5) * 2.0 * 0.6
        res_depth = (self.act(residual_depth) - 0.5) * 2.0 * 2.0 #self.args.disp_to_depth_convert_depth_range
        return res_depth, res_disp, confidence, guide