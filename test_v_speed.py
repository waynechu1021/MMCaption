import triton
from torchvision.models.vision_transformer import EncoderBlock
from models.vit import ViT,Block
#from timm.models.swin_transformer import SwinTransformerBlock
from models.SwinTransformer import SwinTransformerBlock
from models.DeiT import DeiT_Backbone
from models.SwinTransformer import SwinTransformer
from models.vmamba import Backbone_VSSM as vmamba
from models.plainmamba import BackbonePlainMamba,PlainMambaLayer
from models.videomamba import BackboneVideoMamba,create_block
from models.vision_encoder_mamba import Backbone_VSSM,VSSBlock
import torch
import torch.nn as nn
from functools import partial

device = torch.device('cuda:5')
img_size_list =[112*(i+1) for i in range(17)]
# model = Backbone_VSSM(depths = [2, 2, 27, 2],dims = [96, 192, 384, 768],
#         mlp_ratio=0.0,downsample_version='v1',patch_size=4,
#     pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/vssmsmall_dp03_ckpt_epoch_238.pth')
#model = VSSBlock(hidden_dim=768)#BHWC   9M

# model = DeiT_Backbone(img_size=224,patch_size=16,in_chans=3,
#         embed_dim=768,depth=12,num_heads=12,mlp_ratio=4.0,
#         qkv_bias=True,norm_layer = partial(nn.LayerNorm,eps=1e-6),
#         pretrained="/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/deit_base_distilled_patch16_224-df68dfff.pth")

#model = ViT(image_size=224,patch_size=16,num_classes=1000,dim=768,depth=12,mlp_dim=4*768,heads=12)
#model = EncoderBlock(num_heads=12, hidden_dim=768, mlp_dim=int(4.0 * 768),dropout=0.0, attention_dropout=0.0, norm_layer=nn.LayerNorm)#BLC 7M
#model = Block(dim=768,heads=12,dim_head=64,mlp_dim=4*768)

# model = SwinTransformer(depths=[ 2, 2, 18, 2 ],num_heads=[ 4, 8, 16, 32 ],
#                         embed_dim=128,stages=3,img_size=448,
#     pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/swin_base_patch4_window7_224.pth'
#                             )
#model = SwinTransformerBlock(dim=768,input_resolution=(14,14),num_heads=12)#BLC  7M

# model = Backbone_VSSM(depths = [2, 2, 27, 2],dims = [128, 256, 512, 1024],
#         mlp_ratio=0.0,downsample_version='v1',patch_size=4,
#     pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/vssmbase_dp05_ckpt_epoch_260.pth')

# model = BackbonePlainMamba(patch_size=16,embed_dims=448,num_layers=36,stages=35,
#             num_convs_patch_embed= 2,layers_with_dwconv= [],  # useful for L1 model
#             layer_cfgs={'use_rms_norm': False,
#             'mamba_cfg': 
#             {'d_state': 16,
#             'expand': 2,
#             'conv_size': 7,
#             'dt_init': "random",
#             'conv_bias': True,
#             'bias': True,
#             'default_hw_shape': (224 // 16, 224 // 16)
#                 }},
#             pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/plainmamba_l3_new.pth')
# model = PlainMambaLayer(embed_dims=768,use_rms_norm=False,with_dwconv=False,drop_path_rate=0,mamba_cfg={
#                     'd_state': 16,'expand': 4,'conv_size': 7,'dt_init': "random",
#                     'conv_bias': True,'bias': True,'default_hw_shape': (224 // 16, 224 // 16)})#BLC , hw shape   7M

# model = BackboneVideoMamba(depth=32,embed_dim=576,num_frames=8,stages=31,
#                                pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/videomamba_m16_k400_mask_pt_f8_res224.pth'
#                                )
model = create_block(d_model=768)#BLC  4M

model.to(device)
print('params:',sum(p.numel() for p in model.parameters() if p.requires_grad))

forward_list = []
backward_list = []
memory_list = []
for i in range(len(img_size_list)):
    img_size = img_size_list[i]

    # model = SwinTransformerBlock(dim=768,input_resolution=(img_size//16,img_size//16),num_heads=12)#BLC  7M
    # model.to(device)

    torch.cuda.empty_cache()
    inp = torch.randn((1, img_size//16, img_size//16,768)).to(device).requires_grad_()
    inp = inp.reshape(1,-1,768)
    fn = lambda :model(inp)
    ms = triton.testing.do_bench(fn, warmup=100)
    forward_list.append(ms)

    fn = lambda :model(inp)[0].sum().backward()
    ms = triton.testing.do_bench(fn, warmup=100)
    backward_list.append(ms)
    max_memory = torch.cuda.max_memory_allocated(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)
    memory_list.append(max_memory/(1024**3))

print(forward_list)
print(backward_list)
print(memory_list)