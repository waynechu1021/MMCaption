import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from models.multimodal_mamba import init_multimodal_mamba_model
import torch
import triton
import transformers
from train.data import make_supervised_data_module
from dataclasses import dataclass, field
from typing import Optional
from mamba_ssm.models.mixer_seq_simple import _init_weights, MambaLMHeadModel
@dataclass
class MambaConfig:

    d_model: int = 768
    n_layer: int = 24
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
if __name__ == '__main__':
    
    # text = ["what can i say! mamba out!",'But what about tomorrow? Im not sure yet.']
    # gpt_tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2',cache_dir='./pretrained_model/language_model')
    # gpt = transformers.GPT2LMHeadModel.from_pretrained('gpt2',cache_dir='./pretrained_model/language_model')
    
    # print(gpt_tokenizer.sep_token_id)
    # print(gpt_tokenizer.pad_token_id)
    # print(gpt_tokenizer.unk_token_id)
    # print(gpt_tokenizer.eos_token_id)
    # mamba_tokenizer = transformers.AutoTokenizer.from_pretrained('/ssd1/zhuweiye/Multimodal-Mamba/pretrained_model/gpt-neox-20b')
    # config = MambaConfig()
    # mamba = MambaLMHeadModel(config)
    
    # print('params of gpt:',sum(p.numel() for p in gpt.parameters() if p.requires_grad))
    # print('params of mamba:',sum(p.numel() for p in mamba.parameters() if p.requires_grad))

    # device = torch.device('cuda:0')
    # gpt.to(device)
    # mamba.to(device)

    # inp = torch.randint(0,50200,(2, 1024)).to(device)
    # inp2 = inp.detach().to(device)
    # fn = lambda :gpt(inp)
    # ms = triton.testing.do_bench(fn, warmup=100)
    # print(ms)
    # fn = lambda :mamba(inp2)
    # ms = triton.testing.do_bench(fn, warmup=100)
    # print(ms)
    # fn = lambda :gpt(inp)[0].sum().backward()
    # ms = triton.testing.do_bench(fn, warmup=100)
    # print(ms)
    # fn = lambda :mamba(inp2)[0].sum().backward()
    # ms = triton.testing.do_bench(fn, warmup=100)
    # print(ms)

    device = torch.device('cuda:6')
    text_list = [200*(i+1) for i in range(5)]
    #model = transformers.GPT2LMHeadModel.from_pretrained('gpt2',cache_dir='./pretrained_model/language_model')
    #model = transformers.GPT2LMHeadModel.from_pretrained('Multimodal-Mamba/pretrained_model/language_model/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
    config = MambaConfig()
    model = MambaLMHeadModel(config)
    print('params:',sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.to(device)
    forward_list = []
    backward_list = []
    memory_list = []
    for i in range(len(text_list)):
        print(i)
        text_length = text_list[i]
        
        torch.cuda.empty_cache()
        inp = torch.randint(0,50200,(16, text_length)).to(device)

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