import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import time
from tqdm import tqdm

from triton_kernels import triton_deq_int40, triton_q_int40, matmul_q40

INIT_DEVICE = 'meta'
DTYPE = torch.float16
GROUP_SIZE = 64

COMP_TIME = 0.0
DEQ_TIME = 0.0
QKVO_TIME = 0.0
ATTN_TIME = 0.0
MLP_TIME = 0.0
Q_TIME = 0.0
DQ_TIME = 0.0

'''
(Pdb) model.model.config
LlamaConfig {
  "_name_or_path": "./CodeLlama-7b-Instruct-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 16384,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.38.2",
  "use_cache": true,
  "vocab_size": 32016
}
'''
#350-374 MB per batch

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32016
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0

    max_batch_size: int = 80
    max_seq_len: int = 256 # 16384
    max_prompt_seq_len: int = 256

def quantize_q40(w, group_size):
    """
    takes a tensor and returns the Q4_0 quantized version
    i.e. symmetric quantization into int4, [-7, 7]
    """
    #assert w.numel() % group_size == 0
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 7.0
    scale = scale.type(DTYPE)
    scale = scale[:,None]
    # scale into range [-7, 7]
    quant = w / scale
    # round to nearest integer
    
    #assert(quant.max() <= 7)
    #assert(quant.min() >= -7)
    
    int8val = torch.round(quant).to(torch.int8)
    MSB = int8val.view(-1, 2, group_size)[:, 0, :]
    LSB = int8val.view(-1, 2, group_size)[:, 1, :]
    #assert(MSB.abs().max() <= 7)
    #assert(LSB.abs().min() >= -7)
    int8val = (MSB << 4) | (LSB & 0x0F)
    int8val = int8val.view(-1, group_size)

    return int8val, scale #, maxerr

def dequantize_q40(w, scale, group_size, shape, ptdtype: torch.dtype):
    """
    takes a Q4_0 tensor and returns the dequantized version
    """
    # assume it is already packed by group_size
    # w = w.view(-1, group_size)

    MSB = w >> 4
    LSB = w << 4 >> 4 # DO NOT JUST MASK OUT THE MSB, SIGN EXT
    w = torch.hstack((MSB, LSB)).view(-1, group_size)
    # dequantize by rescaling
    fpval = (w * scale).type(ptdtype) #(w * scale.expand(-1, group_size)).type(ptdtype)

    return fpval.view(shape)

#quantize_q40 = triton_q_int40 # outputs gibberish after a while despite tested to be correct?
dequantize_q40 = triton_deq_int40

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return (output * self.weight).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)  # type: ignore
    freqs = torch.outer(t, freqs)  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

'''
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
'''

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, freqs_cis):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = freqs_cis.real.to(DTYPE)
    sin = freqs_cis.imag.to(DTYPE)
    cos = torch.cat((cos, cos), dim = 1)
    sin = torch.cat((sin, sin), dim = 1)
    q = torch.transpose(q, 1, 2)
    k = torch.transpose(k, 1, 2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = torch.transpose(q_embed, 1, 2)
    k_embed = torch.transpose(k_embed, 1, 2)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class LinearQ4_0(torch.nn.Module):
    def __init__(self, in_features, out_features, group_size):
        super().__init__()
        self.group_size = group_size
        self.w = torch.zeros((int(in_features * out_features / group_size / 2), group_size), dtype=torch.int8, device="cpu")
        self.s = torch.zeros(int(in_features * out_features / group_size), dtype=DTYPE, device="cpu")

        self.shape = (in_features, out_features)
        # there must be an external function that init these tensors
        # there are no good ways to do this currently due to pytorch API doesn't save added params that are non-differentiable

    def forward(self, x):
        """
        shape = x.shape
        x=x.view((shape[0]*shape[1], -1)) # first dim is always 0, to optimize kernel we need 2d array
        start = time.time()
        # PyTorch w = w.transpose(), pretransposed in the file weights
        result = matmul_q40(x, self.w, self.s, (self.shape[1], self.shape[0]))
        global COMP_TIME, DEQ_TIME
        COMP_TIME += time.time() - start
        return result.view((shape[0], shape[1], -1))
        """
        start = time.time()
        deq = dequantize_q40(self.w, self.s, self.group_size, self.shape, DTYPE)#.to(self.w.device)
        end = time.time()
        result = F.linear(x, deq)
        global COMP_TIME, DEQ_TIME
        COMP_TIME += time.time() - end
        DEQ_TIME += end - start
        #print(x.shape, "x", self.shape)
        
        #print("DQ/Compute Time: ", (end - start)/(time.time() - end))
        #print("Size in MB: ", torch.prod(torch.Tensor(list(deq.size())))*2/1024/1024)
        return result

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = LinearQ4_0(args.dim, args.n_heads * self.head_dim, GROUP_SIZE)
        self.wk = LinearQ4_0(args.dim, self.n_kv_heads * self.head_dim, GROUP_SIZE)
        self.wv = LinearQ4_0(args.dim, self.n_kv_heads * self.head_dim, GROUP_SIZE)
        self.wo = LinearQ4_0(args.n_heads * self.head_dim, args.dim, GROUP_SIZE)

        promptShape = (1, args.max_prompt_seq_len, self.n_local_kv_heads, self.head_dim)
        cacheShape = (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)
        
        self.cache_prompt_k = torch.zeros(promptShape, dtype=DTYPE).cuda()
        self.cache_prompt_v = torch.zeros(promptShape, dtype=DTYPE).cuda()
        self.cache_prompt_len = 0

        cache_k = torch.zeros(cacheShape, dtype=DTYPE).cuda()
        self.cache_k, self.cache_k_s = quantize_q40(cache_k, GROUP_SIZE)

        cache_v = torch.zeros(cacheShape, dtype=DTYPE).cuda()
        self.cache_v, self.cache_v_s = quantize_q40(cache_v, GROUP_SIZE)
    
    def fork(self, origin: int, curr_batch: int, forks: int):
        '''Forks the process, such that kv-cache is copied from origin batch number to all the list of new batches'''
        self.cache_k = self.cache_k.view((self.args.max_batch_size, -1))
        self.cache_k_s = self.cache_k_s.view((self.args.max_batch_size, -1))
        self.cache_v = self.cache_v.view((self.args.max_batch_size, -1))
        self.cache_v_s = self.cache_v_s.view((self.args.max_batch_size, -1))

        for i in range(forks):
            self.cache_k[curr_batch+i, :] = self.cache_k[origin, :]
            self.cache_k_s[curr_batch+i, :] = self.cache_k_s[origin, :]
            self.cache_v[curr_batch+i, :] = self.cache_v[origin, :]
            self.cache_v_s[curr_batch+i, :] = self.cache_v_s[origin, :]
        self.cache_k = self.cache_k.view((-1, 64))
        self.cache_k_s = self.cache_k_s.view((-1, 64))
        self.cache_v = self.cache_v.view((-1, 1))
        self.cache_v_s = self.cache_v_s.view((-1, 1))

    def resetSeq(self):
        '''Sets sequence length = 0'''
        self.cache_prompt_len = 0
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        global QKVO_TIME, ATTN_TIME, Q_TIME, DQ_TIME
        start = time.time()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        #print("x, xq, xk, xv: ", x.shape, xq.shape, xk.shape, xv.shape)
        QKVO_TIME += time.time() - start
        start = time.time()

        # when start_pos = 0, put to shared prompt KV cache
        # when start_pos != 0, put to regular
        if (start_pos == 0):
            deq_cache_k = xk
            deq_cache_v = xv
            self.cache_prompt_k[:, :seqlen] = xk[0]
            self.cache_prompt_v[:, :seqlen] = xv[0]
            self.cache_prompt_len = seqlen
        else:
            dequant_time = time.time()

            dq_start_pos = start_pos - self.cache_prompt_len
            dq_end_pos = start_pos + seqlen - self.cache_prompt_len
            q_start_pos = dq_start_pos # dq_start_pos // (GROUP_SIZE) * GROUP_SIZE
            q_end_pos = dq_end_pos #((dq_end_pos // (GROUP_SIZE)) + 1) * GROUP_SIZE

            w_seq_block = self.n_local_kv_heads * self.head_dim // 2
            s_seq_block = self.n_local_kv_heads * self.head_dim // 64

            w_shape = (self.args.max_batch_size, self.args.max_seq_len, w_seq_block//64, 64)
            s_shape = (self.args.max_batch_size, self.args.max_seq_len, s_seq_block, 1)
            w_partial_shape = (bsz, q_end_pos - q_start_pos, w_seq_block//64, 64)
            s_partial_shape = (bsz, q_end_pos - q_start_pos, s_seq_block, 1)

            cache_k = self.cache_k.view(w_shape)[:bsz, :q_end_pos].reshape(-1, 64)
            cache_v = self.cache_v.view(w_shape)[:bsz, :q_end_pos].reshape(-1, 64)
            cache_k_s = self.cache_k_s.view(s_shape)[:bsz, :q_end_pos].reshape(-1, 1)
            cache_v_s = self.cache_v_s.view(s_shape)[:bsz, :q_end_pos].reshape(-1, 1)
            #print(cache_k.shape, cache_v.shape, cache_k_s.shape, cache_v_s.shape)
            kv_shape = (bsz, q_end_pos, self.n_local_kv_heads, self.head_dim)
            deq_cache_k = dequantize_q40(cache_k, cache_k_s, GROUP_SIZE, kv_shape, DTYPE)
            deq_cache_v = dequantize_q40(cache_v, cache_v_s, GROUP_SIZE, kv_shape, DTYPE)
            # append curr sequence
            deq_cache_k[:bsz, dq_start_pos : dq_end_pos] = xk
            deq_cache_v[:bsz, dq_start_pos : dq_end_pos] = xv

            DQ_TIME += time.time() - dequant_time
            #print("DQ: ", time.time() - dequant_time)

            quant_time = time.time()
            #self.cache_k, self.cache_k_s = quantize_q40(deq_cache_k, GROUP_SIZE)
            #self.cache_v, self.cache_v_s = quantize_q40(deq_cache_v, GROUP_SIZE)
            
            cache_k, cache_k_s = quantize_q40(deq_cache_k[:bsz, q_start_pos:q_end_pos], GROUP_SIZE)
            cache_v, cache_v_s = quantize_q40(deq_cache_v[:bsz, q_start_pos:q_end_pos], GROUP_SIZE)

            self.cache_k.view(w_shape)[:bsz, q_start_pos:q_end_pos] = cache_k.view(w_partial_shape)
            self.cache_v.view(w_shape)[:bsz, q_start_pos:q_end_pos] = cache_v.view(w_partial_shape)
            self.cache_k_s.view(s_shape)[:bsz, q_start_pos:q_end_pos] = cache_k_s.view(s_partial_shape)
            self.cache_v_s.view(s_shape)[:bsz, q_start_pos:q_end_pos] = cache_v_s.view(s_partial_shape)

            # prepend prompt kv cache after storing the quantized cache
            # print(start_pos, start_pos + seqlen, q_start_pos, q_end_pos)
            deq_cache_k = torch.cat((self.cache_prompt_k.expand(bsz, -1, -1, -1)[:, :self.cache_prompt_len], deq_cache_k[:bsz]), dim=1)
            deq_cache_v = torch.cat((self.cache_prompt_v.expand(bsz, -1, -1, -1)[:, :self.cache_prompt_len], deq_cache_v[:bsz]), dim=1)
            # print(self.cache_prompt_k.shape, self.cache_prompt_k.dtype, deq_cache_k.shape, deq_cache_k.dtype)

            Q_TIME += time.time() - quant_time
            #print("Q", time.time() - quant_time)

        deq_cache_k = deq_cache_k.to(xq)
        deq_cache_v = deq_cache_v.to(xq)

        keys = deq_cache_k[:bsz, : start_pos + seqlen]
        values = deq_cache_v[:bsz, : start_pos + seqlen]
        #print("keys, values: ", keys.shape, values.shape)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        '''
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        print(scores.shape)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        '''
        output = F.scaled_dot_product_attention(xq, keys, values, mask)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        ATTN_TIME += time.time() - start

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = LinearQ4_0(dim, hidden_dim, GROUP_SIZE)
        self.w2 = LinearQ4_0(hidden_dim, dim, GROUP_SIZE)
        self.w3 = LinearQ4_0(dim, hidden_dim, GROUP_SIZE)

    def forward(self, x):
        global MLP_TIME
        start = time.time()
        results = self.w2(F.silu(self.w1(x)) * self.w3(x))
        MLP_TIME += time.time() - start
        return results


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim, device=INIT_DEVICE)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False, device=INIT_DEVICE)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len + self.params.max_prompt_seq_len, 
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        h = h.to(DTYPE)
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen + start_pos), float("-inf"), device=tokens.device
            )
            mask = mask.to(DTYPE).triu(diagonal=start_pos+1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).to(DTYPE)
        return output
    
    def fork(self, origin: int, curr_batch: int, forks: int):
        '''Forks the process, such that kv-cache is copied from origin batch number to all the list of new batches'''
        for layer in self.layers:
            layer.attention.fork(origin, curr_batch, forks)
        return
    
    def resetSeq(self):
        '''Sets all sequence length = 0'''
        for layer in self.layers:
            layer.attention.resetSeq()
        return

    def resetTimers(self):
        global COMP_TIME, DEQ_TIME, QKVO_TIME, ATTN_TIME, MLP_TIME, Q_TIME, DQ_TIME
        COMP_TIME, DEQ_TIME, QKVO_TIME, ATTN_TIME, MLP_TIME, Q_TIME, DQ_TIME = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def printTimers(self):
        print("Linear DQ/Compute Time: ", DEQ_TIME/COMP_TIME)
        print("Times: QKVO_TIME, ATTN_TIME, MLP_TIME, Q_TIME, DQ_TIME, DEQ_TIME, COMP_TIME")
        print("Times:", QKVO_TIME, ATTN_TIME, MLP_TIME, Q_TIME, DQ_TIME, DEQ_TIME, COMP_TIME)
        print("Measured Sum: ", QKVO_TIME+ ATTN_TIME+ MLP_TIME+ Q_TIME+ DQ_TIME+ DEQ_TIME+ COMP_TIME)
    
    @torch.inference_mode()
    def generate(self, idx, batch_size, max_new_tokens, temperature=1.0, top_k=32, enc=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        self.resetTimers()
        self.resetSeq()
        sample_time = 0.0
        model_time = 0.0

        print("Decoding with: batch size =", batch_size)
        assert self.params.max_batch_size >= batch_size
        idx = idx.expand(batch_size, -1)
        print("idx.shape", idx.shape)
        curr_pos = 0
        assert self.params.max_prompt_seq_len >= idx.shape[1]
        max_new_tokens = min(max_new_tokens, self.params.max_seq_len)
        results = torch.zeros(batch_size, max_new_tokens, dtype=torch.int64, device = idx.device)
        results_len = torch.zeros(batch_size, dtype=torch.int32, device = idx.device)
        results_mask = results_len == 0
        print("results.shape", results.shape)
        
        for num_new_tokens in tqdm(range(max_new_tokens)):
            start = time.time()
            #with torch.autograd.profiler.profile(use_cuda=True) as prof2:
            # forward the model to get the logits for the index in the sequence
            if (curr_pos == 0):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.params.max_prompt_seq_len else idx[:, -self.params.max_prompt_seq_len:]
                logits = self(idx_cond[:, curr_pos:], curr_pos)
                curr_pos += idx_cond.shape[1]
            else:
                logits = self(results[:, num_new_tokens-1 : num_new_tokens], curr_pos)
                curr_pos += 1
            #print(prof2.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            #print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            torch.cuda.synchronize()
            model_time += time.time() - start

            start = time.time()
            #with torch.autograd.profiler.profile(use_cuda=True) as prof:
            logits = logits[:, -1, :] # crop to just the final time step
            #print("logits.shape", logits.shape)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature

            # optionally crop the logits to only the top k options
            v, top_k_idx = torch.topk(logits, top_k)
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(v, dim=-1)

            sample = torch.multinomial(probs, num_samples=1, replacement=True)
            idx_next = torch.gather(top_k_idx, 1, sample)
            
            # append sampled index to the running sequence and continue
            # idx = torch.cat((idx, idx_next), dim=1)
            results[:, num_new_tokens] = idx_next[:, 0] * results_mask.int()
            # print(idx_next[:, 0]) 
            
            ends = (idx_next[:, 0] == 2)
            results_len += (~ends).int() * results_mask * 1
            #print(results_mask.int(), results_len, (~ends).int() * 1)
            results_mask = ~((ends) | ~results_mask)
            sample_time += time.time() - start

            #if enc is not None:
            #    print(enc(results, skip_special_tokens = True)[0])
            
            if (torch.sum(~results_mask) == batch_size):
                self.printTimers()
                print("Time (sample, model, multinomial): ", sample_time, model_time)
                return results[:, :num_new_tokens], results_len
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        self.printTimers()
        print("Times (sample, model, multinomial): ", sample_time, model_time)
        return results, results_len

'''
Without cumulative quantization
Decoding with: batch size = 64
idx.shape torch.Size([64, 221])
results.shape torch.Size([64, 291])
100%|██████████████████████████████████████████████████████████| 291/291 [03:27<00:00,  1.40it/s]
DQ/Compute Time:  11.461928811152198
Times: 69.29099774360657 24.072654485702515 10.86605191230774 55.32253670692444 1.3593225479125977
Total time:  207.96499156951904
Tokens per second:  89.5534896850586

With cumulative quantization
Decoding with: batch size = 64
idx.shape torch.Size([64, 221])
results.shape torch.Size([64, 291])
100%|███████████████████████████████████████████████████████████████| 291/291 [01:59<00:00,  2.43it/s]
DQ/Compute Time:  2.698160972897488
Times: 66.64218807220459 7.055298566818237 5.361096382141113 10.561564683914185 17.657522439956665
Total time:  119.92436504364014
Tokens per second:  155.2977294921875
'''
