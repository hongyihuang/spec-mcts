import torch
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from hf_model import Transformer, ModelArgs
from transformers import CodeLlamaTokenizer
import torch.nn.functional as F

DEVICE = "cuda"
DTYPE = torch.float16
GROUP_SIZE = 64

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 7.0 #127.0
    scale = scale.type(DTYPE)
    scale = scale[:,None]
    # scale into range [-127, 127]
    quant = w / scale
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    int8val = int8val.view(-1, group_size)

    # print(int8val.max(), int8val.min())
    assert(int8val.max() <= 7.0)
    assert(int8val.min() >= -7.0)

    return int8val, scale #, maxerr

def dequantize_q80(w, scale, group_size, shape, ptdtype):
    """
    takes a Q8_0 tensor and returns the float version
    """
    # assume it is already packed by group_size
    # w = w.view(-1, group_size)

    # dequantize by rescaling
    return (w * scale).reshape(shape)

def quantize_q40(w, group_size):
    """
    takes a tensor and returns the Q4_0 quantized version
    i.e. symmetric quantization into int4, [-7, 7]
    """
    assert w.numel() % group_size == 0
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
    MSB = int8val.reshape(-1, 2, group_size)[:, 0, :]
    LSB = int8val.reshape(-1, 2, group_size)[:, 1, :]
    assert(MSB.abs().max() <= 7)
    assert(LSB.abs().min() >= -7)
    int8val = (MSB << 4) | (LSB & 0x0F)
    int8val = int8val.view(-1, group_size)

    return int8val, scale #, maxerr

def dequantize_q40(w, scale, group_size, shape, ptdtype):
    """
    takes a Q4_0 tensor and returns the dequantized version
    """
    # assume it is already packed by group_size
    # w = w.view(-1, group_size)

    MSB = w >> 4
    LSB = w << 4 >> 4 # DO NOT JUST MASK OUT THE MSB, SIGN EXT
    w = torch.hstack((MSB, LSB)).view(-1, group_size)

    # dequantize by rescaling
    fpval = (w * scale)#.type(ptdtype)
    return fpval.reshape(shape)

'''
Name Map:

original        | new
================|================
attention_norm  | input_layernorm
ffn_norm        | post_attention_layernorm
feed_forward.w1 | mlp.gate_proj
feed_forward.w2 | mlp.down_proj
feed_forward.w3 | mlp.up_proj
attention.wq    | self_attn.q_proj
attention.wk    | self_attn.k_proj
attention.wv    | self_attn.v_proj
attention.wo    | self_attn.o_proj
norm            | norm
output          | lm_head
tok_embeddings  | embed_tokens
'''

nameMap = {"attention_norm": "input_layernorm",
           "ffn_norm": "post_attention_layernorm",
           "feed_forward": "mlp",
           "w1": "gate_proj",
           "w2": "down_proj",
           "w3": "up_proj",
           "attention": "self_attn",
           "wq": "q_proj",
           "wk": "k_proj",
           "wv": "v_proj",
           "wo": "o_proj",
           "norm": "norm",
           "output": "lm_head",
           "tok_embeddings": "embed_tokens"}

nameMap_reverse = {v: k for k, v in nameMap.items()}

# init from a model saved in a specific directory
def remap_names(file):
    model_dict = torch.load(file, map_location='cpu', mmap=True)
    unwanted_prefix = 'model.'

    for k,v in list(model_dict.items()):
        if k.startswith(unwanted_prefix):
            model_dict[k[len(unwanted_prefix):]] = model_dict.pop(k)

    for k,v in list(model_dict.items()):
        split_keys = k.split(".")
        for i in range(len(split_keys)):
            if split_keys[i] in nameMap_reverse:
                split_keys[i] = nameMap_reverse[split_keys[i]]
        model_dict[".".join(split_keys)] = model_dict.pop(k)
    
    #for k,v in list(model_dict.items()):
    #    model_dict[k] = v.to(torch.float16)
    
    return model_dict

model_dir = './CodeLlama-70b-Instruct-hf/'
dir = os.listdir(model_dir)
# access all {x}-of-00003.bin files
print(dir)
model_dict = {}
for file in dir:
    if file.startswith("pytorch_model-"):
        print("Loading file: ", model_dir + file)
        curr_dict = remap_names(model_dir + file)

        for k,v in list(curr_dict.items()):
            keys = k.split(".")
            if (keys[0] == "layers"):
                layer = int(keys[1]) # layer number
                if (keys[2] == "attention"):
                    if (keys[3] == "wq" or keys[3] == "wk" or keys[3] == "wv" or keys[3] == "wo"):
                        curr_dict.pop(k)
                        k = ".".join(keys[:4])
                        w, s = quantize_q40(v.t(), GROUP_SIZE)
                        curr_dict[k + "." + "w"] = w
                        curr_dict[k + "." + "s"] = s
                        curr_dict[k + "." + "shape"] = v.shape
                elif (keys[2] == "feed_forward"):
                    if (keys[3] == "w1" or keys[3] == "w2" or keys[3] == "w3"):
                        curr_dict.pop(k)
                        k = ".".join(keys[:4])
                        w, s = quantize_q40(v.t(), GROUP_SIZE)
                        curr_dict[k + "." + "w"] = w
                        curr_dict[k + "." + "s"] = s
                        curr_dict[k + "." + "shape"] = v.shape

        for k,v in list(curr_dict.items()):
            if (k.split(".")[-1] == "shape"):
                print(k, v)
            else:
                print(k, v.shape, v.dtype)

        model_dict.update(curr_dict)

breakpoint()
for k,v in list(model_dict.items()):
    if (k.split(".")[-1] == "shape"):
        print(k, v)
    else:
        print(k, v.shape, v.dtype)

torch.save(model_dict, "./spec-mcts/models/llama70b_q40_T.pth")
