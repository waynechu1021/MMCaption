# Implement multimodal MAMBA model
# Inited by mamba_ssm/models/mixer_seq_simple.py

# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
from typing import Union
from collections import namedtuple
import torch.nn.functional as F
import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import _init_weights, MixerModel
# from mamba_ssm.utils.generation import GenerationMixin
from .generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from .vision_encoder_mamba import build_vssm_model
#from .vmamba import build_vssm_model
from .SwinTransformer import build_SwinTransformer_model
from .DeiT import build_DeiT_model
from .plainmamba import build_plainmamba_model
from .videomamba import build_videomamba_model
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from dataclasses import dataclass, field, asdict
from models.hilbert_curve import hibert_index_to_xy
from einops import rearrange
from .mapping_network import *
from timm.models.layers import trunc_normal_
import transformers


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.norm = nn.LayerNorm(in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + residual

def build_vision_encoder(config,**kwargs):
    if config.model_type == "swin":
        return build_SwinTransformer_model(config,**kwargs)
    elif config.model_type == "vmamba":
        return build_vssm_model(config,**kwargs)
    elif config.model_type =='deit':
        return build_DeiT_model(config,**kwargs)
    elif config.model_type == 'plainmamba':
        return build_plainmamba_model(config,**kwargs)
    elif config.model_type == 'videomamba':
        return build_videomamba_model(config,**kwargs)

class MultiModalMixerModel(MixerModel):
    def __init__(self, d_model: int, n_layer: int, vocab_size: int, ssm_cfg=None, norm_epsilon: float = 0.00001, rms_norm: bool = False, initializer_cfg=None, fused_add_norm=False, residual_in_fp32=False, device=None, dtype=None) -> None:
        super().__init__(d_model, n_layer, vocab_size, ssm_cfg, norm_epsilon, rms_norm, initializer_cfg, fused_add_norm, residual_in_fp32, device, dtype)
        # self.ffns = nn.ModuleList([
        #     FFN(d_model,4*d_model)
        #     for i in range(n_layer)
        # ])
    def forward(self, input_ids, vision_feature=None, inference_params=None):
        if vision_feature is not None:
          assert len(vision_feature.shape) == 3
          B, L, D = vision_feature.shape
          assert B == input_ids.shape[0]

        hidden_states = self.embedding(input_ids)
        #hidden_states = input_ids

        if vision_feature is not None:
          hidden_states = torch.concat((vision_feature, hidden_states), dim=1)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        # for layer,ffn in zip(self.layers,self.ffns):
        #     hidden_states, residual = layer(
        #         hidden_states, residual, inference_params=inference_params
        #     )
        #     hidden_states = ffn(hidden_states)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


@dataclass
class MultiModalMambaConfig(MambaConfig):
    model_type: str = 'vmamba'
    patch_size: int = 4
    in_chans: int = 3
    depths: Union[int, list[int]] = field(default=12) #[2, 2, 9, 2]
    dims: Union[int, list[int]] = field(default_factory=96)#[96, 192, 384, 768]
    stages: Union[int, list[int]] = field(default=3)
    d_state: int = 16
    dt_rank: str = 'auto'
    ssm_ratio: float = 2.0
    attn_drop_rate: float = 0.
    drop_rate: float = 0.
    drop_path_rate: float = 0.1
    mlp_ratio: float = 0.0
    patch_norm: bool = True
    norm_layer: nn.Module = nn.LayerNorm
    downsample_version: str = "v1"
    use_checkpoint: bool = False
    out_indices: tuple = (0, 1, 2, 3)
    #used for swin transformer/DeiT
    img_size:int = 224
    embed_dim: int = 128
    num_heads:Union[int, list[int]] = field(default=12)
    window_size:int = 7
    qk_scale:float = None
    qkv_bias:bool = True
    ape:bool = False
    patch_norm:bool = True


    def to_json_string(self) -> str:
        # Convert the dataclass instance to a dict
        class_dict = asdict(self)
            
        # Handle non-serializable fields
        class_dict['norm_layer'] = str(class_dict['norm_layer'].__class__.__name__)
            
        # Serialize the dictionary to a JSON-formatted string
        return json.dumps(class_dict, indent=4)
    
    def to_dict(self) -> dict:
        # Convert the dataclass instance to a dict, handling non-serializable fields manually
        class_dict = asdict(self)
        class_dict['norm_layer'] = str(class_dict['norm_layer'].__class__.__name__)
        return class_dict  
    

class MultiModalMambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MultiModalMambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
        enable_hilbert_scan=False,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.enable_hilbert_scan = enable_hilbert_scan
        # from .SwinTransformer import SwinTransformer
        # self.teacher = SwinTransformer(depths=[ 2, 2, 18, 2 ],
        #                     num_heads=[ 4, 8, 16, 32 ],
        #                     embed_dim=128,
        #                     stages=3)
        self.vision_encoder = build_vision_encoder(config, **factory_kwargs)
        if isinstance(config.stages,int):
            # self.vision_projection_layer = MLP((config.dims[config.stages],config.dims[config.stages]*49//2,d_model))
            self.vision_projection_layer = GLUEncoder(config.dims[config.stages],d_model)
            # self.vision_projection_layer = QueryMamba(config.dims[config.stages],d_model)
            # self.vision_projection_layer = GLUResidual(config.dims[config.stages],d_model)
            # self.vision_projection_layer = Mamba_Qformer(config.dims[config.stages],d_model)
            # self.vision_projection_layer = ImgGLUEncoder(config.dims[config.stages],d_model)
            # self.vision_projection_layer = Qformer(config.dims[config.stages],d_model)
            # self.vision_projection_layer = GatedLinear(config.dims[config.stages],d_model,196 if config.model_type=='deit' else 49)
            # self.vision_projection_layer = CrossAttn(config.dims[config.stages],d_model,196 if config.model_type=='deit' else 49)
            # self.vision_projection_layer = visual2one(config.dims[config.stages],d_model)
            # self.vision_projection_layer = LDP(config.dims[config.stages],d_model)
            # self.vision_projection_layer = nn.Sequential(nn.Linear(config.dims[config.stages], d_model,**factory_kwargs),
            #                                            ScanMerge(d_model))
            # self.vision_projection_layer = nn.Sequential(nn.Linear(config.dims[config.stages], d_model,**factory_kwargs),
            #                                            KBA(d_model))
            # self.vision_projection_layer = nn.Sequential(nn.Linear(config.dims[config.stages], d_model,**factory_kwargs),
            #                                            LearnablePE(196 if config.model_type=='deit' else 49,d_model),)
            # self.vision_projection_layer = nn.Sequential(nn.Linear(config.dims[config.stages], d_model,**factory_kwargs),
            #                                            PosCNN(d_model,d_model))
            # self.vision_projection_layer = nn.Linear(config.dims[config.stages], d_model,**factory_kwargs)
            # self.vision_projection_layer = nn.Linear(config.dims, d_model,**factory_kwargs)
        else:
            self.vision_projection_layer = nn.ModuleList([nn.Linear(config.dims[i], d_model, **factory_kwargs) for i in config.stages])
            self.vision_fusion_layer = GLUEncoder(d_model,d_model)
        
        self.backbone = MultiModalMixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)    
        
        # self.backbone = transformers.GPT2LMHeadModel.from_pretrained('gpt2',cache_dir='./pretrained_model/language_model')
        #self.backbone = transformers.GPT2LMHeadModel.from_pretrained('pretrained_model/language_model/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
        
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, images, image_ids=None, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        if images is None:
            vision_feature = None
        else:
            if isinstance(self.config.stages,int):
                #vision_feature = self.teacher_proj(self.vision_encoder(images)[0])
                #self.vision_feature = vision_feature
                vision_feature = self.vision_projection_layer(self.vision_encoder(images)[0])
                #vision_feature_t = self.vision_projection_layer(self.teacher(images)[0])
            else:
                vision_feature = []
                stage_feature = self.vision_encoder(images)
                for i in range(len(self.config.stages)):
                    vision_feature.append(self.vision_projection_layer[i](stage_feature[i]))
                vision_feature = torch.cat(vision_feature,dim=1)
                vision_feature = self.vision_fusion_layer(vision_feature)
            if self.enable_hilbert_scan:
                W = int(math.sqrt(vision_feature.shape[1]))
                vision_feature = rearrange(vision_feature, 'b (w h) c -> b w h c', w=W, h=W)
                hibert_index = torch.tensor([hibert_index_to_xy(i, W, W) for i in range((W) * (W))])           
                vision_feature = vision_feature[:, hibert_index[:, 0], hibert_index[:, 1]]

        hidden_states = self.backbone(input_ids, vision_feature if images is not None else None, inference_params=inference_params)

        #teach_loss = nn.functional.mse_loss(vision_feature,vision_feature_t) if images is not None else None
        #vision_attn = self.vision_projection_layer(self.vision_feature if images is None else vision_feature,self.backbone.embedding(input_ids))
        #hidden_states = self.backbone(vision_attn,None, inference_params=inference_params)

        # if vision_feature is not None:
        #     input_embed = torch.cat((vision_feature, self.backbone.transformer.wte(input_ids)), dim=1)
        # else:
        #     input_embed = self.backbone.transformer.wte(input_ids)
        # hidden_states = self.backbone(inputs_embeds=input_embed)['logits']
        
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MultiModalMambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        ckpt = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        # _ckpt = {}
        # for k,v,in ckpt.items():
        #     if 'attn_mask' not in k:
        #         _ckpt[k] = v
        msg = model.load_state_dict(ckpt,strict=False)
        print(msg)
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f)


def init_multimodal_mamba_model(vision_encoder_config_path, mamba_lm_model_path, device, dtype):
    import json
    with open(vision_encoder_config_path, 'r') as f:
        vision_config = json.load(f)
    vision_encoder_pretrained = vision_config['pretrained']
    del vision_config['pretrained']
    # create a config file from vision config and lm config
    lm_config = load_config_hf(mamba_lm_model_path)
    multimodal_mamba_config = {**vision_config, **lm_config}
    config = MultiModalMambaConfig(**multimodal_mamba_config)

    # Initialize model
    model = MultiModalMambaLMHeadModel(config, device=device, dtype=dtype)
    # merge weight and load
    lm_weight = load_state_dict_hf(mamba_lm_model_path, device=device, dtype=dtype)
    backbone_weights = {key.replace('backbone.', ''): value for key, value in lm_weight.items() if 'backbone' in key}
    incompatibleKeys = model.backbone.load_state_dict(backbone_weights,strict=False)
    print(incompatibleKeys)
    lm_head_weights = {key.replace('lm_head.', ''): value for key, value in lm_weight.items() if 'lm_head' in key}
    incompatibleKeys = model.lm_head.load_state_dict(lm_head_weights,strict=False)
    print(incompatibleKeys)
    model.vision_encoder.load_pretrained(vision_encoder_pretrained)
    # ckpt = '/ssd1/zhuweiye/Multimodal-Mamba/output-seed/swin-b-mmamba130m-gluencoder-8query/checkpoint-64000/pytorch_model.bin'
    # _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
    # print(f"Successfully load ckpt {ckpt} for teacher model")
    # new_ckpt = {k[15:]:v for k,v in _ckpt.items() if 'vision_encoder' in k}
    # incompatibleKeys = model.teacher.load_state_dict(new_ckpt, strict=False)
    # print(incompatibleKeys)
    # for name, p in model.teacher.named_parameters():
    #     p.requires_grad = False
    return model
