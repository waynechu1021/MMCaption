from einops import rearrange
from timm.layers.norm_act import LayerNormAct2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from collections import OrderedDict
from typing import Tuple
from mamba_ssm.modules.mamba_simple import Mamba,Block

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = x.new_empty((B, 2, C, H * W))
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
            # xs = x.new_empty((B, 2, C, L))
            # xs[:, 0] = x
            # xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            # xs = xs.view(B, 2, C, H, W)
            return xs, None, None

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class KBAFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, att, selfk, selfg, selfb, selfw):
        B, nset, H, W = att.shape
        KK = selfk ** 2
        selfc = x.shape[1]

        att = att.reshape(B, nset, H * W).transpose(-2, -1)

        ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset = selfk, selfg, selfc, KK, nset
        ctx.x, ctx.att, ctx.selfb, ctx.selfw = x, att, selfb, selfw

        bias = att @ selfb
        attk = att @ selfw

        uf = torch.nn.functional.unfold(x, kernel_size=selfk, padding=selfk // 2)# batch, c*winsize*winsize, num_of_window

        # for unfold att / less memory cost   num_of_window = H*W
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)#  batch, g, c//g*kk, num_of_window      
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)# batch, winsize*winsize, g, c//g, c//g*kk

        x = attk @ uf.unsqueeze(-1) 
        del attk, uf
        x = x.squeeze(-1).reshape(B, H * W, selfc) + bias
        x = x.transpose(-1, -2).reshape(B, selfc, H, W)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, att, selfb, selfw = ctx.x, ctx.att, ctx.selfb, ctx.selfw
        selfk, selfg, selfc, KK, nset = ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset

        B, selfc, H, W = grad_output.size()

        dbias = grad_output.reshape(B, selfc, H * W).transpose(-1, -2)

        dselfb = att.transpose(-2, -1) @ dbias.float()
        datt = dbias.float() @ selfb.transpose(-2, -1)

        attk = att @ selfw
        uf = F.unfold(x, kernel_size=selfk, padding=selfk // 2)
        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        dx = dbias.view(B, H * W, selfg, selfc // selfg, 1)

        dattk = dx @ uf.view(B, H * W, selfg, 1, selfc // selfg * KK)
        duf = attk.transpose(-2, -1) @ dx.float()
        del attk, uf

        dattk = dattk.view(B, H * W, -1)
        datt += dattk.float() @ selfw.transpose(-2, -1)
        dselfw = att.transpose(-2, -1) @ dattk.float()

        duf = duf.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx = F.fold(duf, output_size=(H, W), kernel_size=selfk, padding=selfk // 2)

        datt = datt.transpose(-1, -2).view(B, nset, H, W)

        return dx, datt, None, None, dselfb, dselfw

class KBA(nn.Module):
    def __init__(self,dim, group_channel=2, nset=16, k=3):
        super(KBA,self).__init__()
        self.dim = dim
        self.g = self.dim // group_channel
        self.k = k
        self.nset = nset
        self.w = nn.Parameter(torch.zeros(1, nset, self.dim * self.dim // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, self.dim))
        self.init_p(self.w, self.b)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1,groups=self.dim),
        )
        # self.dwconv = nn.Sequential(
        #     nn.Conv2d(dim, self.dim, kernel_size=1),
        #     nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1,
        #               groups=self.dim),
        # )
        interc = min(dim, 16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )
        self.conv211 = nn.Conv2d(in_channels=dim, out_channels=self.nset, kernel_size=1)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.ga1 = nn.Parameter(torch.zeros((1, self.dim, 1, 1)) + 1e-2, requires_grad=True)
        #self.project_out = nn.Conv2d(self.dim, dim, kernel_size=1)

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self,x):
        #x1 = self.dwconv(x)
        # B,L,D = x.shape
        # H = int(math.sqrt(L))
        # assert H**2 == L
        # x = x.permute(0,2,1).reshape(B,D,H,H)
        enhanced_x = self.conv1(x)
        attn = self.conv2(x) * self.attgamma + self.conv211(x)
        # KBA with weighted skip-connection
        x2 = self.KBA(enhanced_x, attn, self.k, self.g, self.b, self.w) * self.ga1 + enhanced_x
        #out = F.gelu(x1)*x2
        #out = self.project_out(out)
        out = x2
        out = out + x
        return out#.reshape(B,D,L).permute(0,2,1)

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x):
        B, N, C = x.shape
        feat_token = x
        H = int(math.sqrt(N))
        assert H**2 == N
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, H)  # reshape
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

class LearnablePE(nn.Module):
    def __init__(self, embed_len=49, embed_dim=768):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
    
    def forward(self,x):
        return x + self.pos_emb

class ScanMerge(nn.Module):
    def __init__(self,input_dim) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=input_dim, out_channels=4, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
    def forward(self,x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        assert H**2 == N
        x = x.transpose(1, 2).view(B, C, H, H)
        alpha = self.fc(x).view(B,4,1,1)#B,4
        x = CrossScan.apply(x)# B,4,C,H*W
        x = alpha*x
        x = torch.sum(x,dim=1)# B,C,H*W
        return x.transpose(1,2)


class LDP(nn.Module):
    def __init__(self,input_dim,hidden_dim) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,3,1,1,groups=hidden_dim),
            LayerNormAct2d(hidden_dim),
            nn.Hardswish(),
            nn.Conv2d(hidden_dim,hidden_dim,1),
            LayerNormAct2d(hidden_dim)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,3,3,1,groups=hidden_dim),
            LayerNormAct2d(hidden_dim),
            nn.Hardswish(),
            nn.Conv2d(hidden_dim,hidden_dim,1),
            LayerNormAct2d(hidden_dim)
        )
    
    def forward(self,x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        assert H**2 == N
        x = self.mlp(x)
        x = x.transpose(1, 2).view(B, C, H, H)
        x = x + self.block1(x)
        x = self.block2(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class visual2one(nn.Module):
    def __init__(self,input_dim,out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
    def forward(self,x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        assert H**2 == N
        x = x.transpose(1, 2).view(B, C, H, H)
        x = self.fc(x).view(B,-1).unsqueeze(1)
        return x

class SelfAttn(nn.Module):
    def __init__(self, embed_dim, num_head=8) -> None:
        super().__init__()
        assert embed_dim % num_head == 0
        self.num_head = num_head
        self.q_fc = nn.Linear(embed_dim,embed_dim)
        self.k_fc = nn.Linear(embed_dim,embed_dim)
        self.v_fc = nn.Linear(embed_dim,embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.fc_out = nn.Linear(embed_dim,embed_dim)
    
    def forward(self,q,k,v):
        B,L,C = q.shape
        _,N,_ = k.shape
        q = self.q_fc(q).reshape(B,L,self.num_head,-1).transpose(1,2)
        k = self.k_fc(k).reshape(B,N,self.num_head,-1).transpose(1,2)
        v = self.v_fc(v).reshape(B,N,self.num_head,-1).transpose(1,2)
        dk = v.shape[-1] ** 0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) / dk
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout1(attn)
        attn = torch.matmul(attn, v).permute(0, 2, 1, 3).flatten(2)
        return self.dropout2(self.fc_out(attn))
    
class BertLayer(nn.Module):
    def __init__(self,embed_dim,layer_id) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attn = SelfAttn(embed_dim)
        if self.layer_id % 2 == 0:
            self.crossattn = SelfAttn(embed_dim)
        self.Intermediate = nn.Sequential(nn.Linear(embed_dim,embed_dim),
                                          nn.Dropout(0.1))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim,4*embed_dim),
                                 nn.GELU(),
                                 nn.Linear(4*embed_dim,embed_dim),
                                 nn.Dropout(0.1))
        self.norm2 = nn.LayerNorm(embed_dim)
                                 
    def forward(self,x,hidden_states):
        hidden_states = self.attn(hidden_states,hidden_states,hidden_states)
        if self.layer_id % 2 == 0:
            hidden_states = self.crossattn(hidden_states,x,x)
        hidden_states = self.norm1(self.Intermediate(hidden_states) + hidden_states)
        hidden_states = self.norm2(self.ffn(hidden_states) + hidden_states)
        return x,hidden_states


class CrossAttn(nn.Module):
    def __init__(self,input_dim,d_model,num_img_tokens):
        super().__init__()
        self.proj_x = nn.Linear(input_dim,d_model)
        self.attn = SelfAttn(d_model)
    
    def forward(self,x,hidden_states):
        x = self.proj_x(x)
        hidden_states = self.attn(hidden_states,x,x)
        return hidden_states

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class GatedLinear(nn.Module):
    def __init__(self,input_dim,d_model,num_img_tokens):
        super().__init__()
        self.attn = SelfAttn(d_model)
        self.alpha = nn.Parameter(torch.rand(1,1,d_model)*(0.999-0.9)+0.9)
        self.fc0 = nn.Linear(d_model,d_model)
        self.fc1 = nn.Linear(d_model,d_model)
        self.fc2 = nn.Linear(d_model,d_model)
        self.fc3 = nn.Linear(d_model,d_model)
        self.fc_out = nn.Linear(d_model,d_model)
        self.norm = RMSNorm(d_model)
    
    def forward(self,x,hidden_states):
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        gate = nn.functional.gelu(self.fc3(hidden_states))
        hidden_states = self.fc0(hidden_states)
        x = self.attn(hidden_states,x,x)#B,L,C
        r = nn.functional.sigmoid(self.fc1(hidden_states))
        i = nn.functional.sigmoid(self.fc2(hidden_states))
        
        alpha = torch.exp(-8*nn.functional.softplus(self.alpha)*r)
        
        hidden_states = alpha * x + torch.sqrt(1-alpha**2)*hidden_states*i
        hidden_states = hidden_states * gate
        return hidden_states + residual


class Qformer(nn.Module):
    def __init__(self, input_dim,d_model,num_query_token=8) -> None:
        super().__init__()
        self.num_query_token = num_query_token
        self.d_model = d_model
        self.query = nn.Parameter(torch.rand(1, num_query_token, d_model))
        self.layers = nn.ModuleList([BertLayer(d_model,i) for i in range(12)])
        self.proj_x = nn.Linear(input_dim,d_model)
    def forward(self,x):
        x = self.proj_x(x)
        hidden_states = self.query.expand(x.shape[0],-1,-1)
        for layer in self.layers:
            x,hidden_states = layer(x,hidden_states)
        return hidden_states


class GLULayer(nn.Module):
    def __init__(self,d_model,layer_id):
        super().__init__()
        self.attn = SelfAttn(d_model)
        self.pe = PosCNN(d_model,d_model)
        #self.conv = nn.Conv1d(d_model,d_model,num_img_tokens)
        self.alpha = nn.Parameter(torch.rand(1,1,d_model)*(0.999-0.9)+0.9)
        
        self.fc = nn.Linear(d_model,d_model)
        self.fc_r = nn.Linear(d_model,d_model)
        self.fc_i = nn.Linear(d_model,d_model)
        self.fc_gate = nn.Linear(d_model,d_model)
        self.fc_out = nn.Linear(d_model,d_model)
        self.norm = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.mlp_gate = nn.Linear(d_model,3*d_model)
        self.mlp_fc = nn.Linear(d_model,3*d_model)
        self.out = nn.Linear(3*d_model,d_model)
    # def alpha_init(self,w: torch.Tensor):
    #     min_rad = 0.9
    #     max_rad = 0.999
    #     eps = 1e-6
    #     with torch.no_grad():
    #         # Proportional to area in a ring.
    #         # 0.5 * jnp.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + 1e-8)
    #         w.uniform_(min_rad ** 2 + eps, max_rad ** 2 + eps)
    #         w.log_().mul_(0.5)
    #         w.neg_().exp_().sub_(1.0).log_()
    #         return w
    
    def forward(self,x,hidden_states):
        #x-img feature
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        gate = nn.functional.silu(self.fc_gate(hidden_states))
        hidden_states = self.fc(hidden_states)
        x = self.pe(x)
        x = self.attn(hidden_states,x,x)#B,L,C
        # x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        r = nn.functional.sigmoid(self.fc_r(hidden_states))
        i = nn.functional.sigmoid(self.fc_i(hidden_states))
        
        alpha = torch.exp(-8*nn.functional.softplus(self.alpha)*r)
        
        hidden_states = alpha * x + torch.sqrt(1-alpha**2)*hidden_states*i
        hidden_states = hidden_states * gate + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        gate = nn.functional.silu(self.mlp_gate(hidden_states))
        hidden_states = self.mlp_fc(hidden_states)
        hidden_states = self.out(hidden_states * gate) + residual
        return hidden_states 

class GLUEncoder(nn.Module):
    def __init__(self,input_dim,d_model,num_query_token=8):
        super().__init__()
        self.proj_x = nn.Linear(input_dim,d_model)
        self.layers = nn.ModuleList([GLULayer(d_model,i) for i in range(12)])
        self.query = nn.Parameter(torch.rand(1, num_query_token, d_model))
    def forward(self,x):
        x = self.proj_x(x)#B, L, C
        hidden_states = self.query.expand(x.shape[0],-1,-1)

        # B, N, C = x.shape
        # H = int(math.sqrt(N))
        # assert H**2 == N
        # x = x.transpose(1, 2).view(B, C, H, H)
        # x = CrossScan.apply(x).permute(0,1,3,2)#B,4,C,H*W - B,4,H*W,C
        #shift_size = x.shape[1]//len(self.layers)
        for i,layer in enumerate(self.layers):
            hidden_states = layer(x,hidden_states)
            #x = torch.roll(x,shift_size,dims=1)
        return hidden_states

class ImgGLULayer(nn.Module):
    def __init__(self,d_model,layer_id,num_img_tokens=49):
        super().__init__()
        #self.conv = nn.Conv1d(d_model,d_model,num_img_tokens)
        self.alpha = nn.Parameter(torch.rand(1,1,d_model)*(0.999-0.9)+0.9)
        self.fc = nn.Linear(d_model,d_model)
        self.fc_r = nn.Linear(d_model,d_model)
        self.fc_i = nn.Linear(d_model,d_model)
        self.fc_gate = nn.Linear(d_model,d_model)
        self.fc_out = nn.Linear(d_model,d_model)
        self.norm = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.mlp_gate = nn.Linear(d_model,3*d_model)
        self.mlp_fc = nn.Linear(d_model,3*d_model)
        self.out = nn.Linear(3*d_model,d_model)
    
    def forward(self,hidden_states):
        B,L,C = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        gate = nn.functional.gelu(self.fc_gate(hidden_states))
        hidden_states = self.fc(hidden_states)

        r = nn.functional.sigmoid(self.fc_r(hidden_states))
        i = nn.functional.sigmoid(self.fc_i(hidden_states))
        
        alpha = torch.exp(-8*nn.functional.softplus(self.alpha)*r)
        out = (torch.sqrt(1-alpha[:,0]**2)*hidden_states[:,0]*i[:,0]).unsqueeze(1)
        for h in range(1,L):
            h_next = alpha[:,h] * out[:,h-1] + torch.sqrt(1-alpha[:,h]**2)*hidden_states[:,h]*i[:,h]
            out = torch.cat((out,h_next.unsqueeze(1)),dim=1)
        hidden_states = out * gate + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        gate = nn.functional.gelu(self.mlp_gate(hidden_states))
        hidden_states = self.mlp_fc(hidden_states)
        hidden_states = self.out(hidden_states * gate) + residual
        return hidden_states 

class ImgGLUEncoder(nn.Module):
    def __init__(self,input_dim,d_model):
        super().__init__()
        self.proj_x = nn.Linear(input_dim,d_model)
        #self.layers = nn.ModuleList([ImgGLULayer(d_model,i) for i in range(12)])
        self.layers = nn.ModuleList([nn.Sequential(*[ImgGLULayer(d_model,i) for i in range(12)]) for j in range(4)])
    def forward(self,x):
        x = self.proj_x(x)

        B, N, C = x.shape
        H = int(math.sqrt(N))
        assert H**2 == N
        x = x.transpose(1, 2).view(B, C, H, H)
        x = CrossScan.apply(x).permute(0,1,3,2)#B,4,C,H*W - B,4,H*W,C

        # for i,layer in enumerate(self.layers):
        #     x = layer(x)

        out = []
        for i in range(4):
            x_i = x[:,i]
            x_i = self.layers[i](x_i)
            out.append(x_i)
        x = torch.stack(out,dim=1)
        x = CrossMerge.apply(x.permute(0,1,3,2).reshape(B,4,C,H,H)).flatten(2).transpose(1,2)
        return x

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)



class Mamba_Qformer(nn.Module):
    def __init__(self,input_dim,d_model,num_query_token=7):
        super().__init__()
        self.num_query_token = num_query_token
        self.proj_x = nn.Linear(input_dim,d_model)
        # self.query = nn.Parameter(torch.rand(1, num_query_token, 1, d_model))
        self.query = nn.Parameter(torch.rand(1, num_query_token, d_model))
        self.layers = nn.ModuleList([Block(dim=d_model,mixer_cls=Mamba) for i in range(32)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self,hidden_states):
        hidden_states = self.proj_x(hidden_states)
        B,L,C = hidden_states.shape
        # assert L % self.num_query_token == 0
        # hidden_states = hidden_states.reshape(B,self.num_query_token,-1,C)
        # hidden_states = torch.cat((hidden_states,self.query.expand(B,-1,-1,-1)),dim=2)
        # hidden_states = hidden_states.reshape(B,-1,C)
        hidden_states = torch.cat((hidden_states,self.query.expand(B,-1,-1)),dim=1)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        # hidden_states = hidden_states.reshape(B,self.num_query_token,-1,C)
        # hidden_states = hidden_states[:,:,-1]
        hidden_states = hidden_states[:,-self.num_query_token:]
        return hidden_states

class GLUResidual(GLUEncoder):
    def __init__(self,input_dim,d_model,num_query_token=8):
        super().__init__(input_dim,d_model,num_query_token)
        self.conv = nn.Sequential(nn.Conv2d(3,d_model,4,4),
                                  LayerNormAct2d(d_model),
                                  #KBA(d_model)
                                  nn.Conv2d(d_model,d_model,3,2,1),
                                  LayerNormAct2d(d_model),
                                  KBA(d_model),
                                  )
    def forward(self,x, img):
        x = self.proj_x(x)#B, L, C
        residual = self.conv(img).flatten(2).permute(0,2,1)
        x = torch.cat((x,residual),dim=1)
        hidden_states = self.query.expand(x.shape[0],-1,-1)

        for i,layer in enumerate(self.layers):
            hidden_states = layer(x,hidden_states)
        return hidden_states

from timm.layers import trunc_normal_,DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn,mamba_inner_fn
from einops import repeat
from causal_conv1d import causal_conv1d_fn

class QueryMambaOp(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.query_proj = nn.Linear(self.d_model,self.d_inner,bias=True, **factory_kwargs)

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
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
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
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states,query):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        conv_state, ssm_state = None, None
        query = self.query_proj(query)
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            #x_dbl = self.x_proj(rearrange(query.permute(0,2,1), "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                #z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            if self.layer_idx % 2 == 0:
                y = y * self.act(query)
            out = self.out_proj(y)
        return out

class QueryMambaLayer(nn.Module):
    def __init__(
        self,
        embed_dims=768,
        use_rms_norm=False,
        drop_path_rate=0.,
        layer_idx = 0,
    ):
        super(QueryMambaLayer, self).__init__()

        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)
            self.norm_query = nn.LayerNorm(embed_dims)

        self.mamba = QueryMambaOp(embed_dims,layer_idx=layer_idx)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x,query):
        B,L,C = x.shape
        mixed_x = self.drop_path(self.mamba(self.norm(x),self.norm_query(query)))
        mixed_x = mixed_x + x
        return mixed_x

class QueryMamba(nn.Module):
    def __init__(self, input_dim = 768,d_model=768,num_query_tokens = 49):
        super().__init__()
        #self.proj = nn.Linear(input_dim,d_model)
        self.query = nn.Parameter(torch.rand(1,num_query_tokens,d_model))
        self.layers = nn.ModuleList([QueryMambaLayer(embed_dims=d_model,layer_idx=i) for i in range(24)])
        trunc_normal_(self.query, std=0.02)
    
    def forward(self,x):
        #x = self.proj(x)
        B,L,C = x.shape
        query = self.query.expand(B,-1,-1)
        for i,layer in enumerate(self.layers):
            query = layer(x,query)
        return query

if __name__ == "__main__":
    #model = Mamba_Qformer(768,768)# 12layer param 45M  32layer param 121M
    #model = GLUEncoder(768,768)#12layer param 128M
    #model = Qformer(768,768)#12layer param 106M
    model = GLULayer(768,0)# 12layer 59M 24layer 118M
    device = torch.device('cuda:5')
    model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.rand(2,49,768).to(device)
    y = model(x)
    print(y.shape)