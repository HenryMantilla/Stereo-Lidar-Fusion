import torch
import torch.nn as nn

from models.encoders.PVT import PvtFamily
from models.modules import CBAM
from models.dyspn import Dynamic_deformablev2

class DepthFusionPVT(nn.Module):
    def __init__(self, convnext_pretrained):
        super().__init__()

        # Embedding layers
        self.rgb_embed = ResidualBlock(in_channels=3, out_channels=32)
        self.lidar_embed = ResidualBlock(in_channels=1, out_channels=16)

        # PVT Backbone
        pvt_family = PvtFamily()
        self.pvt = pvt_family.initialize_model(
            config_name="pvt_medium", patch_size=4, in_chans=48, num_stages=4
        )

        # Main ConvDecoder
        self.conv_decoder = ConvDecoder(in_channels=512, out_channels=50)

        # Flexible Decoders for intermediate features of ConvDecoder
        self.flexible_decoders = None
        self.dyspn_modules = None

    def initialize_flexible_decoders(self, main_decoder_features):
        """
        Initialize FlexibleDecoder dynamically based on the channels of main_decoder_features.
        """
        self.flexible_decoders = nn.ModuleList([
            FlexibleDecoder(in_channels=feature.size(1), out_channels=50, num_upsample=4 - i).cuda()
            for i, feature in enumerate(main_decoder_features)
        ])
        self.dyspn_modules = nn.ModuleList([
            Dynamic_deformablev2(iteration=6).cuda() for _ in range(len(self.flexible_decoders))
        ])


    def forward(self, rgb, stereo, lidar):
        # Embedding layers
        rgb_emb = self.rgb_embed(rgb)
        lidar_emb = self.lidar_embed(stereo)

        vit_input = torch.cat([rgb_emb, lidar_emb], dim=1)
        out_pvt, intermediate_pvt = self.pvt(vit_input, vit_input)
        out_decoder, main_decoder_features = self.conv_decoder(out_pvt, intermediate_pvt)

        if self.flexible_decoders is None:
            self.initialize_flexible_decoders(main_decoder_features)

        # First pass through FlexibleDecoder
        flexible_outputs = [decoder(feature) for decoder, feature in zip(self.flexible_decoders, main_decoder_features)]

        refinement_output = None
        intermediate_refinement = []

        for i, (dyspn, flexible_output) in enumerate(zip(self.dyspn_modules, flexible_outputs)):
            # Prepare inputs for dyspn
            depth_input = flexible_output[:, 49:50, :, :] + (refinement_output if refinement_output is not None else stereo) #lidar
            #depth_input = torch.nan_to_num(depth_input, nan=0.0, posinf=1e6, neginf=-1e6)
            guide1 = flexible_output[:, 0:24, :, :]  # Guide 1
            guide2 = flexible_output[:, 24:48, :, :]  # Guide 2
            confidence = flexible_output[:, 48:49, :, :]  # Confidence

            # Process through dyspn
            refinement_output = dyspn(depth_input, guide1, guide2, confidence, lidar)
            intermediate_refinement.append(refinement_output)

        return depth_input, refinement_output, confidence, intermediate_refinement


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        """
        Residual block with optional 1x1 convolution for channel matching.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        
        # Main branch
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        # Shortcut branch
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

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
        def upsample_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
                CBAM(out_ch)  
            )

        self.stage1 = upsample_conv(in_channels, 512)   
        self.stage2 = upsample_conv(512+320, 256)               
        self.stage3 = upsample_conv(256+128, 128)             
        self.stage4 = upsample_conv(128+64, 64)  
        self.stage5 = upsample_conv(64, out_channels)

    def forward(self, x, feats):
        intermediate_features = []

        x = self.stage1(x)
        intermediate_features.append(x)

        x = torch.cat([x, feats[2]], dim=1)
        x = self.stage2(x)
        intermediate_features.append(x)

        x = torch.cat([x, feats[1]], dim=1)
        x = self.stage3(x)
        intermediate_features.append(x)

        x = torch.cat([x, feats[0]], dim=1)
        x = self.stage4(x)
        intermediate_features.append(x)

        x = self.stage5(x)
        intermediate_features.append(x)

        return x, intermediate_features

class FlexibleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_upsample=1, kernel_size=2, stride=2, padding=0):
        super(FlexibleDecoder, self).__init__()
        """
        Flexible Decoder block with adjustable transpose convolution parameters for upsampling.
        
        Args:
            in_channels (int): Number of input channels for the feature map.
            out_channels (int): Number of output channels for the feature map.
            num_upsample (int): Number of upsampling layers (for gradual upsampling).
            kernel_size (int or tuple): Kernel size for transpose convolution.
            stride (int or tuple): Stride for transpose convolution.
            padding (int or tuple): Padding for transpose convolution.
        """
        self.num_upsample = num_upsample

        # Initialize upsampling layers with conv transpose + batch norm + activation
        if num_upsample > 0:
            upsampling_layers = []
            for _ in range(num_upsample):
                upsampling_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
                in_channels = out_channels  # Update in_channels for subsequent layers
            self.upsampling = nn.Sequential(*upsampling_layers)
        else:
            self.upsampling = None  # No upsampling layers

    def forward(self, x):
        """
        Forward pass through the decoder block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Decoded output tensor.
        """
        if self.upsampling:
            x = self.upsampling(x)
        return x
