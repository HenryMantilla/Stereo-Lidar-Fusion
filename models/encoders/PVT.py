import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_
from models.modules import ConvMixer, CBAM
from models.modules import PatchEmbedding, SpatialReductionAttention, Mlp


class PvtFamily():
    def __init__(self):

        self.model_configs = {
            "pvt_tiny": {
                "img_size": 224,
                "embed_dims": [64, 128, 320, 512],
                "num_heads": [1, 2, 5, 8],
                "mlp_ratios": [8, 8, 4, 4],
                "qkv_bias": True,
                "norm_layer": partial(nn.LayerNorm, eps=1e-6),
                "depths": [2, 2, 2, 2],
                "sr_ratios": [8, 4, 2, 1],
            },
            "pvt_small": {
                "img_size": 224,
                "embed_dims": [64, 128, 320, 512],
                "num_heads": [1, 2, 5, 8],
                "mlp_ratios": [8, 8, 4, 4],
                "qkv_bias": True,
                "norm_layer": partial(nn.LayerNorm, eps=1e-6),
                "depths": [3, 4, 6, 3],
                "sr_ratios": [8, 4, 2, 1],
            },
            "pvt_medium": {
                "img_size": 224,
                "embed_dims": [64, 128, 320, 512],
                "num_heads": [1, 2, 5, 8],
                "mlp_ratios": [8, 8, 4, 4],
                "qkv_bias": True,
                "norm_layer": partial(nn.LayerNorm, eps=1e-6),
                "depths": [3, 4, 18, 3],
                "sr_ratios": [8, 4, 2, 1],
            },
            "pvt_large": {
                "img_size": 224,
                "embed_dims": [64, 128, 320, 512],
                "num_heads": [1, 2, 5, 8],
                "mlp_ratios": [8, 8, 4, 4],
                "qkv_bias": True,
                "norm_layer": partial(nn.LayerNorm, eps=1e-6),
                "depths": [3, 8, 27, 3],
                "sr_ratios": [8, 4, 2, 1],
            },
            "pvt_huge": {
                "img_size": 224,
                "embed_dims": [128, 256, 512, 768],
                "num_heads": [2, 4, 8, 12],
                "mlp_ratios": [8, 8, 4, 4],
                "qkv_bias": True,
                "norm_layer": partial(nn.LayerNorm, eps=1e-6),
                "depths": [3, 10, 60, 3],
                "sr_ratios": [8, 4, 2, 1],
            },
        }

    def initialize_model(self, config_name, patch_size=4, in_chans=3, num_classes=1000, num_stages=4):
        config = self.model_configs.get(config_name)
        if config is None:
            raise ValueError(f"Configuration '{config_name}' not found.")
        
        model = PyramidVisionTransformer(
            img_size=config["img_size"],
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dims=config["embed_dims"],
            num_heads=config["num_heads"],
            mlp_ratios=config["mlp_ratios"],
            qkv_bias=config["qkv_bias"],
            norm_layer=config["norm_layer"],
            depths=config["depths"],
            sr_ratios=config["sr_ratios"],
            num_stages=num_stages
        )
        return model

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialReductionAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.cbam = CBAM(dim)
        #self.convmixer_layer = ConvMixer(dim, kernel_size=7, depth=2, groups=dim//4)
        self.conv = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)


    def forward(self, x_stereo, x_sparse, H, W):

        x = x_sparse + self.drop_path(self.attn(self.norm1(x_stereo), self.norm1(x_sparse), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        B, L, C = x.shape
        x_reshaped = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        #x_cnn = self.convmixer_layer(x_reshaped)
        x_cbam = self.cbam(x_reshaped)

        x = torch.cat([x_reshaped, x_cbam], dim=1)
        x = self.conv(x)
        x = x.permute(0,2,3,1).view(B,L,C)

        return x
    
class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbedding(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x_stereo, x_sparse):
        B = x_stereo.shape[0]

        intermediate_features = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            
            x_stereo, (H, W) = patch_embed(x_stereo)
            x_sparse, _ = patch_embed(x_sparse)

            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x_stereo = pos_drop(x_stereo + pos_embed)
            x_sparse = pos_drop(x_sparse + pos_embed)

            for blk in block:
                x_sparse = blk(x_stereo, x_sparse, H, W)
            
            intermediate_features.append(x_sparse.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
                
            x_stereo = x_stereo.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x_sparse = x_sparse.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return intermediate_features

    def forward(self, x_stereo, x_sparse):
        intermediate_features = self.forward_features(x_stereo, x_sparse)

        return intermediate_features
    

########################
"""
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialReductionAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        in_ch = [144, 288, 640, 1152]

        for i in range(num_stages):
            patch_embed = PatchEmbedding(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            reduce_ch = nn.Conv2d(in_ch[i], embed_dims[i], kernel_size=3, padding=1)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"conv_reduce{i + 1}", reduce_ch)

        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x, cnn_features):
        B = x.shape[0]
        intermediate_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            reduce_ch = getattr(self, f"conv_reduce{i + 1}")
            
            x, (H, W) = patch_embed(x)

            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)

            for blk in block:
                x = blk(x, H, W)
            
                
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = reduce_ch(torch.cat([x, cnn_features[i]], dim=1))

            intermediate_features.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
            
        return x, intermediate_features

    def forward(self, x, cnn_features):
        x, intermediate = self.forward_features(x, cnn_features)

        return x, intermediate
"""