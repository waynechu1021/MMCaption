# Implement mamba based vision encoder
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from typing import Union

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
# try:
#     #from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
#     from selective_scan.selective_scan.selective_scan_interface import selective_scan_fn, selective_scan_ref
# except:
#     pass

# an alternative for mamba_ssm
# try:
#     from selective_scan import selective_scan_fn as selective_scan_fn_v1
#     from selective_scan import selective_scan_ref as selective_scan_ref_v1
# except:
#     pass

if True:
    import selective_scan_cuda_core as selective_scan_cuda
    
    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            # xs = x.new_empty((B, 1, C, H * W))
            # xs[:, 0] = x.flatten(2, 3)
            # xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            return xs
        
        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            # ys = ys[:, 0].unsqueeze(1) + ys[:, 1].flip(dims=[-1]).view(B, 1, -1, L)
            # y = ys[:, 0]
            return y.view(B, -1, H, W)
    
    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            # ys = ys[:, 0].unsqueeze(1) + ys[:, 1].flip(dims=[-1]).view(B, 1, D, -1)
            # y = ys[:, 0]
            return y
        
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            # xs = x.new_empty((B, 1, C, L))
            # xs[:, 0] = x
            # xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            # xs = xs.view(B, 1, C, H, W)
            return xs, None, None

    def cross_selective_scan(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)
        
        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)
        
        return y

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def print_jit_input_names(inputs):
    # tensor.11, dt.1, A.1, B.1, C.1, D.1, z.1, None
    try: 
        print("input params: ", end=" ", flush=True)
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)

    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[1].debugName().startswith("dts") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    assert inputs[4].debugName().startswith("Cs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    # flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


DEV = False
class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        if DEV:
            d_conv = -1
            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 4
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.K2 = self.K 
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    # def forward_corev2(self, x: torch.Tensor, nrows=-1):
    #     return cross_selective_scan(
    #         x, self.x_proj_weight[0].unsqueeze(0), None, self.dt_projs_weight[0].unsqueeze(0), self.dt_projs_bias[0].unsqueeze(0).contiguous(),
    #         self.A_logs[:self.A_logs.shape[0]//4], self.Ds[:self.Ds.shape[0]//4], getattr(self, "out_norm", None), self.softmax_version, 
    #         nrows=nrows,
    #     )
    def forward_corev2(self, x: torch.Tensor, nrows=-1):
        return cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version, 
            nrows=nrows,
        )
    
    # forward_core = forward_core_share_ssm
    # forward_core = forward_core_share_a
    # forward_core = forward_corev1
    forward_core = forward_corev2
    # forward_core = forward_corev0

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x)) # (b, d, h, w)
            y = self.forward_core(x)
            if self.softmax_version:
                y = y * z
            else:
                y = y * F.silu(z)
        else:
            if self.softmax_version:
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                x = F.silu(x)
            else:
                xz = F.silu(xz)
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            y = self.forward_core(x)
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        #shift_size = 0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)
        #self.shift_size = shift_size
    def _forward(self, input: torch.Tensor):
        #input = torch.roll(input,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN B,H,W,C
        #x = torch.roll(x,shifts=(self.shift_size,self.shift_size),dims=(1,2))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
            self, 
            patch_size=4, 
            in_chans=3, 
            num_classes=1000, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            # =========================
            d_state=16, 
            dt_rank="auto", 
            ssm_ratio=2.0, 
            attn_drop_rate=0., 
            softmax_version=False,
            # =========================
            drop_rate=0., 
            drop_path_rate=0.1, 
            mlp_ratio=4.0,
            patch_norm=True, 
            norm_layer=nn.LayerNorm,
            downsample_version: str = "v2",
            use_checkpoint=False,  
            **kwargs,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(self.embed_dim) if patch_norm else nn.Identity()), 
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            if downsample_version == "v2":
                downsample = self._make_downsample(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()
            else:
                downsample = PatchMerging2D(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()
                #downsample = nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                depth = depths[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                attn_drop_rate=attn_drop_rate,
                softmax_version=softmax_version,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        depth=2,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        d_state=16,
        dt_rank="auto",
        ssm_ratio=2.0,
        attn_drop_rate=0.0, 
        softmax_version=False,
        # ===========================
        mlp_ratio=4.0,
        drop_rate=0.0,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
                #shift_size=0 if (d%2==0) else 5,
                **kwargs,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.CrossScan": None,
            "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, patch_size=4, in_chans=3,  
                 depths=[2, 2, 9, 2], dims=[96, 192, 384, 768], 
                 d_state=16, ssm_ratio=2.0, attn_drop_rate=0., 
                 drop_rate=0., drop_path_rate=0.1, mlp_ratio=4.0,
                 patch_norm=True,
                 downsample_version: str = "v1",
                 use_checkpoint=False,
                 pretrained=None, 
                 stages:Union[int,list[int]] = 3,
                 **kwargs,
        ):
        super().__init__(patch_size=patch_size, in_chans=in_chans,
                         depths=depths, dims=dims, 
                         d_state=d_state, ssm_ratio=ssm_ratio, attn_drop_rate=attn_drop_rate, 
                         drop_rate=drop_rate, drop_path_rate=drop_path_rate, mlp_ratio=mlp_ratio,
                         patch_norm=patch_norm, norm_layer=nn.LayerNorm,
                         downsample_version=downsample_version,
                         use_checkpoint=use_checkpoint,
                         **kwargs)
        
        self.stages = [stages] if isinstance(stages,int) else stages
        self.last_stage = stages if isinstance(stages,int) else max(stages)

        del self.classifier
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

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i in range(self.last_stage+1):
            out, x = layer_forward(self.layers[i], x) # (B, H, W, C)
            out = out.flatten(1,2)
            if i in self.stages:
                outs.append(out)
        
        return outs
    
    
def build_vssm_model(config, device='cuda', dtype=torch.float32):
    model = Backbone_VSSM(
        patch_size=config.patch_size, 
        in_chans=config.in_chans, 
        depths=config.depths, 
        dims=config.dims, 
        # ===================
        d_state=config.d_state,
        dt_rank=("auto" if config.dt_rank == "auto" else int(config.dt_rank)),
        ssm_ratio=config.ssm_ratio,
        attn_drop_rate=config.attn_drop_rate,
        # ===================
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        mlp_ratio=config.mlp_ratio,
        patch_norm=config.patch_norm,
        #norm_layer=nn.Identity,#config.norm_layer,
        downsample_version=config.downsample_version,
        use_checkpoint=config.use_checkpoint,
        stages=config.stages
    )
    model = model.to(dtype)
    model = model.to(device)
    return model



if __name__ == "__main__":
    # model = Backbone_VSSM(depths = [2, 2, 27, 2],dims = [96, 192, 384, 768],
    #                       mlp_ratio=0.0,patch_size=4,stages=3,
    #                       pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/vssmsmall_dp03_ckpt_epoch_238.pth'
    #                       )
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # device = torch.device('cuda:7')
    # model.to(device)
    # x = torch.rand(2,3,224,224).to(device)
    # y = model(x)
    # for i in y:
    #     print(i.shape)

    def check_vssblock(device):
        import triton
        from torchvision.models.vision_transformer import EncoderBlock
        from timm.models.vision_transformer import Block
        #from timm.models.swin_transformer import SwinTransformerBlock
        from SwinTransformer import SwinTransformerBlock
        from DeiT import DeiT_Backbone
        from SwinTransformer import SwinTransformer
        from vmamba import Backbone_VSSM as vmamba
        from simba import BackboneSiMBA
        from plainmamba import BackbonePlainMamba
        img_size=224*2

        vb = Backbone_VSSM(depths = [2, 2, 27, 2],dims = [96, 192, 384, 768],
                           mlp_ratio=0.0,
                           downsample_version='v1',
                           patch_size=4,
                           pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/vssmsmall_dp03_ckpt_epoch_238.pth').to(device)
        trans = vmamba(depths = [2, 2, 27, 2],dims = [96, 192, 384, 768],norm_layer='ln',ssm_conv_bias=True,
                          mlp_ratio=0.0,patch_size=4,downsample_version='v1',forward_type="v4",ssm_d_state=16,patchembed_version='v1',
                          pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/vssmsmall_dp03_ckpt_epoch_238.pth').to(device)
        # trans = BackbonePlainMamba(patch_size=16,embed_dims=448,num_layers=36,stages=-1,
        #     num_convs_patch_embed= 2,layers_with_dwconv= [],  # useful for L1 model
        #     layer_cfgs={'use_rms_norm': False,
        #     'mamba_cfg': 
        #     {'d_state': 16,
        #     'expand': 2,
        #     'conv_size': 7,
        #     'dt_init': "random",
        #     'conv_bias': True,
        #     'bias': True,
        #     'default_hw_shape': (224 // 16, 224 // 16)
        #         }},
        #     pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/plainmamba_l3_new.pth').to(device)
        # trans = Backbone_Vim(patch_size=16, stride=8, embed_dim=384,
        #         depth=24, rms_norm=True, residual_in_fp32=True, 
        #         fused_add_norm=True, final_pool_type='all', if_abs_pos_embed=True,
        #         if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, 
        #         if_devide_out=True, use_middle_cls_token=True,
        #         pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/vim_s_midclstok_ft_81p6acc.pth'
        #                 ).to(device)

        # trans = BackboneSiMBA(stem_hidden_dim = 64,
        #                 embed_dims = [64, 128, 320, 512], 
        #                 mlp_ratios = [8, 8, 4, 4], 
        #                 norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        #                 depths = [3, 4, 12, 3], 
        #                 sr_ratios = [4, 2, 1, 1], stages=3).to(device)
        # trans = SwinTransformer(img_size=img_size,depths=[ 2, 2, 18, 2 ],num_heads=[ 4, 8, 16, 32 ],embed_dim=128,
        #                        pretrained='/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/vision_encoder/swin_base_patch4_window7_224.pth'
        #                        ).to(device)
        #trans = SwinTransformerBlock(dim=512,input_resolution=(img_size,img_size),num_heads=16,).to(device)
        # trans = EncoderBlock(
        #     num_heads=16, 
        #     hidden_dim=512, 
        #     mlp_dim=int(4.0 * 16), 
        #     dropout=0.0, 
        #     attention_dropout=0.0, 
        #     norm_layer=nn.LayerNorm,
        # ).to(device)

        print('params of vmamba:',sum(p.numel() for p in vb.parameters() if p.requires_grad))
        print('params of vmamba-speed:',sum(p.numel() for p in trans.parameters() if p.requires_grad))
        inp = torch.randn((4, 3, img_size, img_size)).to(device).requires_grad_()
        inp2 = inp.detach().to(device).requires_grad_()
        fn = lambda :vb(inp)
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        fn = lambda :trans(inp2)
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        fn = lambda :vb(inp)[0].sum().backward()
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        fn = lambda :trans(inp2)[0].sum().backward()
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        import time; time.sleep(2)
    device = torch.device('cuda:0')
    check_vssblock(device)
    
    
