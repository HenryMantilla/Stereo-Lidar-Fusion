import torch.nn as nn

class LidarStereoFusion(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(LidarStereoFusion, self).__init__()

        self.lidar_conv = nn.Sequential(
            ConvBlk(in_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=2),
            ConvBlk(out_channels, out_channels, dilation=4),
            ConvBlk(out_channels, out_channels, dilation=8),
            ConvBlk(out_channels, out_channels, dilation=16),
        )

        self.stereo_conv = nn.Sequential(
            ConvBlk(in_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=2),
            ConvBlk(out_channels, out_channels, dilation=4),
            ConvBlk(out_channels, out_channels, dilation=8),
            ConvBlk(out_channels, out_channels, dilation=16),
        )

        self.fusion_conv = nn.Sequential(
            ConvBlk(out_channels, out_channels, dilation=8),
            ConvBlk(out_channels, out_channels, dilation=4),
            ConvBlk(out_channels, out_channels, dilation=2),
            ConvBlk(out_channels, 1, dilation=1),
        )

    def forward(self, lidar_depth, stereo_depth):

        x1 = self.lidar_conv(lidar_depth)
        x2 = self.stereo_conv(stereo_depth)

        x = x1 + x2
        x = self.fusion_conv(x)

        return x

class Refinement(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(Refinement, self).__init__()

        self.fused_disparity_conv = nn.Sequential(
            ConvBlk(in_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=1),
        )

        self.color_guidance_conv = nn.Sequential(
            ConvBlk(3, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=1),
        )

        self.final_disparity = nn.Sequential(
            ConvBlk(out_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=1),
            ConvBlk(out_channels, out_channels, dilation=1),
            ConvBlk(out_channels, 1, dilation=1),
        )

    def forward(self, fused_disparity, rgb_left):

      x1 = self.fused_disparity_conv(fused_disparity)
      x2 = self.color_guidance_conv(rgb_left)

      x = x1 + x2
      x = self.final_disparity(x)

      return x

class FusionModel(nn.Module):
    def __init__(self, in_channels):
        super(FusionModel, self).__init__()

        self.lidar_stereo_fusion = LidarStereoFusion(in_channels)
        self.refinement = Refinement(in_channels)

    def forward(self, rgb_left, stereo_disp, lidar_disp):

        fused_disparity = self.lidar_stereo_fusion(lidar_disp, stereo_disp)
        refined_disparity = self.refinement(fused_disparity, rgb_left)

        refined_disparity = fused_disparity + refined_disparity

        return fused_disparity, refined_disparity


class ConvBlk(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ConvBlk, self).__init__()

        self.conv_blk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.conv_blk(x)