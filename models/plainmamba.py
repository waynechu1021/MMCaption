from typing import Dict, Tuple, Union,Sequence
import math
from einops import repeat
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import trunc_normal_,DropPath,to_2tuple
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm
import inspect

def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = F.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)

def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        norm_layer = layer_type
    elif layer_type.lower() == 'ln':
        norm_layer = nn.LayerNorm

    abbr = 'ln'

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

class AdaptivePadding(nn.Module):
    """Applies padding adaptively to the input.

    This module can make input get fully covered by filter
    you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad
    zero around input. The "corner"  mode would pad zero
    to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel. Default: 1.
        stride (int | tuple): Stride of the filter. Default: 1.
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".

    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super().__init__()
        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        """Calculate the padding size of input.

        Args:
            input_shape (:obj:`torch.Size`): arrange as (H, W).

        Returns:
            Tuple[int]: The padding size along the
            original H and W directions
        """
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        """Add padding to `x`

        Args:
            x (Tensor): Input tensor has shape (B, C, H, W).

        Returns:
            Tensor: The tensor with adaptive padding
        """
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x

class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 num_convs=0,
                 conv_type='Conv2d',
                 patch_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super(ConvPatchEmbed, self).__init__()

        assert patch_size % 2 == 0

        self.embed_dims = embed_dims
        if stride is None:
            stride = patch_size // 2
        else:
            stride = stride // 2

        self.stem = torch.nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True))

        if num_convs > 0:
            convs = []
            for _ in range(num_convs):
                convs.append(torch.nn.Conv2d(64, 64, (3,3), (1,1), padding=1, bias=False))
                convs.append(torch.nn.BatchNorm2d(64))
                convs.append(torch.nn.ReLU(True))
            self.convs = torch.nn.Sequential(*convs)
        else:
            self.convs = None

        kernel_size = to_2tuple(patch_size//2)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        # self.projection = build_conv_layer(
        #     cfg=dict(
        #     type=conv_type,
        #     in_channels=64,
        #     out_channels=embed_dims,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     bias=bias)

        self.projection = nn.Conv2d(
            in_channels=64,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            _input_size = (input_size[0] // 2, input_size[1] // 2)
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(_input_size)
                input_h, input_w = _input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                _input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (_input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (_input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """
        x = self.stem(x)
        if self.convs is not None:
            x = self.convs(x)

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size

class PlainMamba2D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_size=7,
        conv_bias=True,
        bias=False,
        init_layer_scale=None,
        default_hw_shape=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None
        self.n_directions = 4
        if default_hw_shape is not None:
            orders, inverse_orders, directions = self.get_permute_order(default_hw_shape)
            (
                self.default_permute_order,
                self.default_permute_order_inverse,
                self.default_direction
            ) = orders, inverse_orders, directions

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        padding = int(conv_size // 2)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=(conv_size, conv_size),
            stride=(1, 1),
            padding=(padding, padding),
            groups=self.d_inner
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions+1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)


    def get_permute_order(self, hw_shape):
        if self.default_permute_order is not None:
            if hw_shape[0] == self.default_hw_shape[0] and hw_shape[1] == self.default_hw_shape[1]:
                return self.default_permute_order, self.default_permute_order_inverse, self.default_direction
        H, W = hw_shape
        L = H * W

        # [start, right, left, up, down] [0, 1, 2, 3, 4]

        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        j_d = "right"
        while i < H:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W-1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "left"

            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        o2 = []
        d2 = []
        o2_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H-1, W-1
            j_d = "left"
        else:
            i, j = H-1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d2.append(1)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d2.append(2)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = "right"
        d2 = [0] + d2[:-1]

        o3 = []
        d3 = []
        o3_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o3_inverse[idx] = len(o3)
            o3.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d3.append(4)
                else:
                    j = j + 1
                    d3.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d3.append(3)
                else:
                    j = j + 1
                    d3.append(1)
                    i_d = "down"
        d3 = [0] + d3[:-1]

        o4 = []
        d4 = []
        o4_inverse = [-1 for _ in range(L)]

        if W % 2 == 1:
            i, j = H - 1, W - 1
            i_d = "up"
        else:
            i, j = 0, W - 1
            i_d = "down"
        while j > -1:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o4_inverse[idx] = len(o4)
            o4.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d4.append(4)
                else:
                    j = j - 1
                    d4.append(2)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d4.append(3)
                else:
                    j = j - 1
                    d4.append(2)
                    i_d = "down"
        d4 = [0] + d4[:-1]

        o1 = tuple(o1)
        d1 = tuple(d1)
        o1_inverse = tuple(o1_inverse)

        o2 = tuple(o2)
        d2 = tuple(d2)
        o2_inverse = tuple(o2_inverse)

        o3 = tuple(o3)
        d3 = tuple(d3)
        o3_inverse = tuple(o3_inverse)

        o4 = tuple(o4)
        d4 = tuple(d4)
        o4_inverse = tuple(o4_inverse)

        return (o1, o2, o3, o4), (o1_inverse, o2_inverse, o3_inverse, o4_inverse), (d1, d2, d3, d4)

    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        conv_state, ssm_state = None, None

        xz = self.in_proj(x)  # [B, L, 2 * E]
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=-1)
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + d_state * 2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        dt = dt.permute(0, 2, 1).contiguous()  # [B, d_innter, L]
        B = B.permute(0, 2, 1).contiguous()  # [B, d_state, L]
        C = C.permute(0, 2, 1).contiguous()  # [B, d_state, L]

        assert self.activation in ["silu", "swish"]

        orders, inverse_orders, directions = self.get_permute_order(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]  # each [L, d_state]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in direction_Bs]
        ys = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),
                dt,
                A,
                (B + dB).contiguous(),
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_o, :]
            for o, inv_o, dB in zip(orders, inverse_orders, direction_Bs)
        ]
        y = sum(ys) * self.act(z)
        out = self.out_proj(y)

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out


class PlainMambaLayer(nn.Module):
    def __init__(
        self,
        embed_dims,
        use_rms_norm,
        with_dwconv,
        drop_path_rate,
        mamba_cfg,
    ):
        super(PlainMambaLayer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})

        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    embed_dims,
                    embed_dims,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    bias=False,
                    groups=embed_dims
                ),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )
        self.mamba = PlainMamba2D(**mamba_cfg)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, hw_shape):
        mixed_x = self.drop_path(self.mamba(self.norm(x), hw_shape))
        mixed_x = mixed_x + x
        if self.with_dwconv:
            b, l, c = mixed_x.shape
            h, w = hw_shape
            mixed_x = mixed_x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            mixed_x = self.dw(mixed_x)
            mixed_x = mixed_x.reshape(b, c, h * w).permute(0, 2, 1)
        return mixed_x
    
class PlainMamba(nn.Module):
    arch_zoo = {
        'L1': {
            'patch_size': 16,
            'embed_dims': 192,
            'num_layers': 24,
            'num_convs_patch_embed': 1,
            'layers_with_dwconv': [0],  # useful for L1 model
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (224 // 16, 224 // 16)
                }
            }
        },
        'L2': {
            'patch_size': 16,
            'embed_dims': 384,
            'num_layers': 24,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (224 // 16, 224 // 16)
                }
            }
        },
        'L3': {
            'patch_size': 16,
            'embed_dims': 448,
            'num_layers': 36,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (224 // 16, 224 // 16)
                }
            }
        },
    }

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 arch=None,
                 patch_size=16,
                 embed_dims=192,
                 num_layers=20,
                 num_convs_patch_embed=1,
                 with_pos_embed=True,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 interpolate_mode='bicubic',
                 layer_cfgs=dict(),
                 layers_with_dwconv=[],
                 init_cfg=None,
                 test_cfg=dict(),
                 convert_syncbn=False,
                 freeze_patch_embed=False,
                 **kwargs):
        super(PlainMamba, self).__init__()

        self.test_cfg = test_cfg

        self.img_size = to_2tuple(img_size)
        self.convert_syncbn = convert_syncbn

        self.arch = arch
        if self.arch is None:
            self.embed_dims = embed_dims
            self.num_layers = num_layers
            self.patch_size = patch_size
            self.num_convs_patch_embed = num_convs_patch_embed
            self.layers_with_dwconv = layers_with_dwconv
            _layer_cfgs = layer_cfgs
        else:
            assert self.arch in self.arch_zoo.keys()
            self.embed_dims = self.arch_zoo[self.arch]['embed_dims']
            self.num_layers = self.arch_zoo[self.arch]['num_layers']
            self.patch_size = self.arch_zoo[self.arch]['patch_size']
            self.num_convs_patch_embed = self.arch_zoo[self.arch]['num_convs_patch_embed']
            self.layers_with_dwconv = self.arch_zoo[self.arch]['layers_with_dwconv']
            _layer_cfgs = self.arch_zoo[self.arch]['layer_cfgs']

        self.with_pos_embed = with_pos_embed
        self.interpolate_mode = interpolate_mode
        self.freeze_patch_embed = freeze_patch_embed
        _drop_path_rate = drop_path_rate

        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            num_convs=self.num_convs_patch_embed,
            patch_size=self.patch_size,
            stride=self.patch_size
        )

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        if with_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
            trunc_normal_(self.pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # if isinstance(out_indices, int):
        #     out_indices = [out_indices]
        # assert isinstance(out_indices, Sequence), \
        #     f'"out_indices" must by a sequence or int, ' \
        #     f'get {type(out_indices)} instead.'
        # for i, index in enumerate(out_indices):
        #     if index < 0:
        #         out_indices[i] = self.num_layers + index
        #     assert 0 <= out_indices[i] <= self.num_layers, \
        #         f'Invalid out_indices {index}'
        # self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, _drop_path_rate, self.num_layers)
        self.drop_path_rate = _drop_path_rate

        self.layer_cfgs = _layer_cfgs

        self.layers = nn.ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [copy.deepcopy(_layer_cfgs) for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            _layer_cfg_i = layer_cfgs[i]
            _layer_cfg_i.update({
                "embed_dims": self.embed_dims,
                "drop_path_rate": dpr[i]
            })
            if i in self.layers_with_dwconv:
                _layer_cfg_i.update({"with_dwconv": True})
            else:
                _layer_cfg_i.update({"with_dwconv": False})
            self.layers.append(
                PlainMambaLayer(**_layer_cfg_i)
            )

        self.final_norm = final_norm
        # assert final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        # for i in out_indices:
        #     if i != self.num_layers - 1:
        #         if norm_cfg is not None:
        #             norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
        #         else:
        #             norm_layer = nn.Identity()
        #         self.add_module(f'norm_layer{i}', norm_layer)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super(PlainMamba, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.with_pos_embed:
                trunc_normal_(self.pos_embed, std=0.02)
        self.set_freeze_patch_embed()

    def set_freeze_patch_embed(self):
        if self.freeze_patch_embed:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.with_pos_embed:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0
            )
            x = x + pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)


            B, _, C = x.shape
            patch_token = x
            # if i != self.num_layers - 1:
            #     norm_layer = getattr(self, f'norm_layer{i}')
            #     patch_token = norm_layer(patch_token)
            #patch_token = patch_token.permute(0, 3, 1, 2)
            outs.append(patch_token)
        return outs

class BackbonePlainMamba(PlainMamba):
    def __init__(self, img_size=224, in_channels=3, arch=None, 
                 patch_size=16, embed_dims=192, num_layers=20, 
                 num_convs_patch_embed=1, with_pos_embed=True, 
                 drop_rate=0, drop_path_rate=0, norm_cfg=dict(type='LN', eps=0.000001), 
                 final_norm=True, interpolate_mode='bicubic', layer_cfgs=dict(), layers_with_dwconv=[], 
                 init_cfg=None, test_cfg=dict(), convert_syncbn=False, freeze_patch_embed=False, 
                 pretrained=None, stages:Union[int,list[int]] = 36,**kwargs):
        super().__init__(img_size, in_channels, arch, patch_size, 
                         embed_dims, num_layers, num_convs_patch_embed, 
                         with_pos_embed, drop_rate, drop_path_rate,
                           norm_cfg, final_norm, interpolate_mode, layer_cfgs,
                             layers_with_dwconv, init_cfg, test_cfg, convert_syncbn,
                               freeze_patch_embed, **kwargs)
        
        self.stages = [stages] if isinstance(stages,int) else stages
        for i, stage in enumerate(self.stages):
            if stage < 0:
                self.stages[i] = self.num_layers + stage
        self.last_stage = stages if isinstance(stages,int) else max(stages)

        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="key"):
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
        x, patch_resolution = self.patch_embed(x)

        if self.with_pos_embed:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0
            )
            x = x + pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            B, _, C = x.shape
            patch_token = x
            if i in self.stages:
                outs.append(patch_token)
        return outs

def build_plainmamba_model(config, device='cuda', dtype=torch.float32):
    model = BackbonePlainMamba(
        patch_size=config.patch_size, 
        num_layers=config.depths, 
        embed_dims=config.dims, 
        # ===================
        num_convs_patch_embed=2,
        layer_cfgs={'use_rms_norm': False,
            'mamba_cfg': 
            {'d_state': 16,
            'expand': 2,
            'conv_size': 7,
            'dt_init': "random",
            'conv_bias': True,
            'bias': True,
            'default_hw_shape': (224 // 16, 224 // 16)
                }},
        stages=config.stages
    )
    model = model.to(dtype)
    model = model.to(device)
    return model

if __name__ == "__main__":
    model = BackbonePlainMamba(patch_size=16,embed_dims=448,num_layers=36,stages=35,
            num_convs_patch_embed= 2,layers_with_dwconv= [],  # useful for L1 model
            layer_cfgs={'use_rms_norm': False,
            'mamba_cfg': 
            {'d_state': 16,
            'expand': 2,
            'conv_size': 7,
            'dt_init': "random",
            'conv_bias': True,
            'bias': True,
            'default_hw_shape': (224 // 16, 224 // 16)
                }},
            pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/plainmamba_l3_new.pth')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    device = torch.device('cuda:6')
    model.to(device)
    x = torch.rand(2,3,448,448).to(device)
    y = model(x)
    for i in y:
        print(i.shape)

    # _ckpt = torch.load(open('/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/plainmamba_l3.pth', "rb"), map_location=torch.device("cpu"))
    # _ckpt = _ckpt['state_dict']
    # new_ckpt = {}
    # for k,v in _ckpt.items():
    #     if 'backbone' in k:
    #         k = k[9:]
    #     new_ckpt[k] = v
    # _ckpt = {'key':new_ckpt}
    # torch.save(_ckpt,'/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/plainmamba_l3_new.pth')