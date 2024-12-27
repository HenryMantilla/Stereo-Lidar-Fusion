import torch
import torch.nn as nn
from models.modules import ConvexUpsampling, CBAM, conv_block
    
class ConvDecoderWithConvexUp(nn.Module):
    def __init__(self, in_chans, feature_channels=[256, 128, 64, 32, 16], out_chans=1):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(len(feature_channels)):
            fc = feature_channels[i]
            in_ch = in_chans[i]

            block = nn.Sequential(
                ConvexUpsampling(in_ch, upsample_factor=2),
                conv_block(in_ch, fc, 3, 1, 1),
                CBAM(fc)
            )

            self.blocks.append(block)

        self.final_conv = nn.Sequential(
            ConvexUpsampling(feature_channels[-1], upsample_factor=2),
            conv_block(feature_channels[-1], out_chans, 1),
            nn.ReLU(),
            )

    def forward(self, cnn_feat, vit_out):
        
        cnnf_1, cnnf_2, cnnf_3, cnnf_4 = cnn_feat
        block_1, block_2, block_3, block_4, block_5 = self.blocks

        x_1 = block_1(vit_out)
        x_2 = block_2(torch.cat((cnnf_4, x_1), dim=1))
        x_3 = block_3(torch.cat((cnnf_3, x_2), dim=1))
        x_4 = block_4(torch.cat((cnnf_2, x_3), dim=1))
        x_5 = block_5(torch.cat((cnnf_1, x_4), dim=1))

        return self.final_conv(x_5)


class ConvDecoder(nn.Module):
    def __init__(self, in_ch, n_features, kernel, stride, padding, out_padding):
        super().__init__()

        out_features = [n_features * 4, n_features * 2, n_features, n_features]

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_ch if i == 0 else in_ch[i-1], ch, kernel, stride, padding, out_padding),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ) for i, ch in enumerate(out_features)
        ])

        self.final_layer = nn.Sequential(nn.ConvTranspose2d(n_features, 1, kernel, stride, padding, out_padding),
                                         nn.Sigmoid())
        
    def forward(self, x):

        for layer in self.decoder:
            x = layer(x)

        x = self.final_layer(x)

        return x