import math
import struct
import inspect
import time

from Config import LLMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# RMSNorm层: 实现均方根归一化
# 相比于 LayerNorm，少算一个均值，而且效果差不多
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)).type_as(x)
    

def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6, ):
    # 计算频率因子
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成时间步长向量
    t = torch.arange(end, device = freqs.device)
    # 计算外积，得到每个时间步对应的角度
    freqs = torch.outer(t, freqs).float()
    # 将幅度固定为 1，通过极坐标得到复数表示
    pos_cis = torch.polar(torch.ones_like(freqs), freqs) # complex64
    return pos_cis

def apply_rotary_emb(xq, xk, pos_cis):
    # 定义辅助函数，调整 pos_cis 的形状以匹配输入张量
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # 确保 pos_cis 的形状为 (序列长度, head维度)
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)
    # 将 xq 和 xk 转换为复数形式，方便与位置编码相乘
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    # 应用旋转位置编码后再转换为实数表示，并展平最后一维
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# 复制 KV 头
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[ : , : , : , None, : ]
        .expand(bs, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: LLMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0, "Error!"
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // self.n_heads
        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal = 1)
        self.register_buffer('mask', mask, persistent = False)
        
    def forward(self, 
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        # kv_cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim = 1)
            xv = torch.cat([past_key_value[1], xv], dim = 1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 实际使用 mask 时，只取需要的那部分
        scores += self.mask[ : , : , : seq_len, : seq_len]
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = scores @ xv
        
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        # 没有设置 MLP 的维度，那就用 dim * 8 / 3
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias = False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias = False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias = False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class XiaoXinBlock(nn.Module):
    def __init__(self, layer_id: int, config: LLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps = config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps = config.norm_eps)
        self.feed_forward = FeedForward(config)
        
    def forward(self,
                x,
                pos_cis,
                past_key_value = None,
                use_cache = False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value = past_key_value,
            use_cache = use_cache
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


class XiaoXin(PreTrainedModel):
    config_class = LLMConfig
    
    def __init__(self, params: LLMConfig = None):
        self.params = params or LLMConfig()
        super().__init__(params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([
            XiaoXinBlock(l, params) for l in range(self.n_layers)
        ])
        self.norm = RMSNorm(params.dim, eps = params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias = False)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer("pos_cis",
                            precompute_pos_cis(dim = params.dim // params.n_heads, theta = params.rope_theta), persistent = False)
        self.OUT = CausalLMOutputWithPast()

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args
                ):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)
        
        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []

        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h,
                pos_cis,
                past_key_value = past_key_values[l], 
                use_cache = use_cache
            )
            past_kvs.append(past_kv)
        
        logits = self.output(self.norm(h))
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('past_key_values', past_kvs)

        return self.OUT
    
    @torch.inference_mode() # 不参与梯度计算，加速推理过程
    def generate(self, input_ids, eos_token_id = 2, max_new_tokens = 1024, temperature = 0.75, top_p = 0.90, stream = False, rp = 1., use_cache = True, pad_token_id = 0, **args):
        # 生成文本的主函数
        # input_ids: 输入的 token id (形状为 [batch_size, seq_len])
        # eos_token_id: 结束符的 token id (默认为 2)
        # max_new_tokens: 生成的最大 token 数量 (默认为 1024)
        # temperature: 多样性控制温度
        # top_p: 控制 nucleus采样 的阈值
        # stream: 是否进行流式生成
        # rp: 重复惩罚因子 (默认为 1.0)
        # use_cache: 是否使用缓存 (默认为 True)
        # pad_token_id: padding token 的 id
        return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        # 流式生成的实现方法
        # 生成过程分步进行，每次生成一个 token
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                # 如果是  第一次生成 or 不使用缓存，从头开始生成
                # 这里的 self 调用的是 forward
                out, first_seq = self(input_ids, past_key_values = past_kvs, use_cache = use_cache, ** args), False
            else:
                # 否则，只使用最后一个 token 进行生成
                out = self(input_ids[ : , -1:], past_key_values = past_kvs, use_cache = use_cache, start_pos = input_ids.shape[1] - 1, **args)

            # 获取 logits 和 past_key_values
            logits, past_kvs = out.logits[ : , -1, : ], out.past_key_values
            
            # 对已生成的 tokens 进行重复惩罚
            logits[ : , list(set(input_ids.tolist()[0]))] /= rp

            # 对 logits 进行温度调整
            logits /= (temperature + 1e-9)

            if top_p is not None and top_p < 1.0:
                # 如果 top_p 不为 None 且小于 1， 则应用 nucleus 采样
                sorted_logits, sorted_indices = torch.sort(logits, descending = True, dim = -1)
                sorted_probs = F.softmax(sorted_logits, dim = -1)
                cumulative_probs = torch.cumsum(sorted_probs, dim = -1)
                # 选择累计概率超过 top_p 的 token 进行过滤
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[ : , 1 : ] = sorted_indices_to_remove[ : , : -1].clone()
                sorted_indices_to_remove[ : , 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf') # 过滤掉不符合条件的 token
            
            # 从 logits 中根据概率分布采样下一个 token
            input_ids_next = torch.multinomial(F.softmax(logits, dim = -1), num_samples = 1)

            # 将新生成的 token 添加到 input_ids 中
            input_ids = torch.cat((input_ids, input_ids_next), dim = 1)

            # 输出当前生成的 token 
            yield input_ids[ : , start: ]

            # 如果生成的 token 为 eos_token，则停止生成
            if input_ids_next.item() == eos_token_id:
                break

            
