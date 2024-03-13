import torch
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from model import Transformer, ModelArgs
from transformers import CodeLlamaTokenizer
import torch.nn.functional as F

DEVICE = "cpu"
DTYPE = torch.float16
GROUP_SIZE = 64

# start by running in int8, then move on to int4
# file = "./CodeLlama-7b-Instruct-hf/pytorch_model-00003-of-00003.bin"
# model_dict = torch.load(file, map_location='cpu', mmap=True)

# print(model_dict.keys())
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

model_dir = './CodeLlama-7b-Instruct-hf/'
dir = os.listdir(model_dir)
# access all {x}-of-00003.bin files
print(dir)
model_dict = {}
for file in dir:
    if file.startswith("pytorch_model-"):
        print("Loading file: ", model_dir + file)
        curr_dict = remap_names(model_dir + file)
        model_dict.update(curr_dict)


#for k,v in list(model_dict.items()):
    #print(k, v.shape, v.dtype)
    #w, s = quantize_q40(v, 64)

    #dequant = dequantize_q40(w, s, 64, v.shape)
    #print("Avg error: ", torch.mean(torch.abs(v - dequant)))

model = Transformer(ModelArgs) #default is llama7B
model.load_state_dict(model_dict, strict=False, assign=True)

model.eval()

tokenizer = CodeLlamaTokenizer.from_pretrained("./CodeLlama-7b-Instruct-hf")

PROMPT = '''# Write a python function to find the longest chain which can be formed from the given set of pairs.

# Test Cases 
assert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3
assert max_chain_length([Pair(1, 2), Pair(3, 4),Pair(5, 6), Pair(7, 8)], 4) == 4
assert max_chain_length([Pair(19, 10), Pair(11, 12),Pair(13, 14), Pair(15, 16), Pair(31, 54)], 5) == 5

def max_chain_length(pairs, n):
    
'''
model.eval()
input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
input_ids = input_ids.to(DEVICE)
print(input_ids)
print("Generating...")

print(tokenizer.batch_decode(input_ids, skip_special_tokens = True)[0])
generated_ids = model.generate(input_ids, 128, temperature=0.2, top_k=32, enc=tokenizer.batch_decode)

print(generated_ids)
print(tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0])

