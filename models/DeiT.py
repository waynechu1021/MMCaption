# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import math
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

class DeiT_Backbone(DistilledVisionTransformer):
    def __init__(self, pretrained=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_embed.strict_img_size = False
        del self.head
        del self.head_dist
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self,x):
        B = x.shape[0]
        x = self.patch_embed(x)
        H = int(math.sqrt(x.shape[1]))

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        if x.shape[1] != self.pos_embed.shape[1]:
            h = int(math.sqrt(self.patch_embed.num_patches))
            pos_embed = self.pos_embed[:,2:].permute(0, 2, 1).reshape(1,self.embed_dim,h,h)
            pos_embed = nn.functional.interpolate(pos_embed,size=H,mode="bicubic",align_corners=True)
            pos_embed = pos_embed.reshape(1,self.embed_dim,-1).permute(0,2,1)
            pos_embed = torch.cat([self.pos_embed[:,:2],pos_embed],dim=1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        #x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return [x[:,2:]]



def build_DeiT_model(config, pretrained=None, device='cuda', dtype=torch.float32):
    model = DeiT_Backbone(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        embed_dim=config.embed_dim,
        depth=config.depths,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        #norm_layer=partial(config.norm_layer,eps=1e-6),
        norm_layer = partial(nn.LayerNorm,eps=1e-6)
        #stages=config.stages
    )
    model = model.to(dtype)
    model = model.to(device)
    return model

if __name__ == "__main__":
    import yaml
    model = DeiT_Backbone()
    device = torch.device('cuda:1')
    model.to(device)
    x = torch.rand(8,3,336,336).to(device)
    y = model(x)
    print(y.shape)