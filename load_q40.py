import torch
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from hf_model_q40 import Transformer, ModelArgs
from transformers import CodeLlamaTokenizer
import torch.nn.functional as F

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
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

model_file = './spec-mcts/models/llama7b_q40.pth'

print(dir)
model_dict = {}
print("Loading file: ", model_file)
curr_dict = remap_names(model_file)
model_dict.update(curr_dict)

for k,v in list(model_dict.items()):
    if (k.split(".")[-1] == "shape"):
        print(k, v)
    else:
        print(k, v.shape, v.dtype)

model = Transformer(ModelArgs) #default is llama7B
model.load_state_dict(model_dict, strict=False, assign=True)

for i, layer in enumerate(model.layers):    
    key = "layers." + str(i)
    layer.attention.wq.w = model_dict[key + ".attention.wq.w"].to(DEVICE)
    layer.attention.wq.s = model_dict[key + ".attention.wq.s"].to(DEVICE)
    layer.attention.wq.shape = model_dict[key + ".attention.wq.shape"]

    layer.attention.wk.w = model_dict[key + ".attention.wk.w"].to(DEVICE)
    layer.attention.wk.s = model_dict[key + ".attention.wk.s"].to(DEVICE)
    layer.attention.wk.shape = model_dict[key + ".attention.wk.shape"]

    layer.attention.wv.w = model_dict[key + ".attention.wv.w"].to(DEVICE)
    layer.attention.wv.s = model_dict[key + ".attention.wv.s"].to(DEVICE)
    layer.attention.wv.shape = model_dict[key + ".attention.wv.shape"]

    layer.attention.wo.w = model_dict[key + ".attention.wo.w"].to(DEVICE)
    layer.attention.wo.s = model_dict[key + ".attention.wo.s"].to(DEVICE)
    layer.attention.wo.shape = model_dict[key + ".attention.wo.shape"]

    layer.feed_forward.w1.w = model_dict[key + ".feed_forward.w1.w"].to(DEVICE)
    layer.feed_forward.w1.s = model_dict[key + ".feed_forward.w1.s"].to(DEVICE)
    layer.feed_forward.w1.shape = model_dict[key + ".feed_forward.w1.shape"]

    layer.feed_forward.w2.w = model_dict[key + ".feed_forward.w2.w"].to(DEVICE)
    layer.feed_forward.w2.s = model_dict[key + ".feed_forward.w2.s"].to(DEVICE)
    layer.feed_forward.w2.shape = model_dict[key + ".feed_forward.w2.shape"]

    layer.feed_forward.w3.w = model_dict[key + ".feed_forward.w3.w"].to(DEVICE)
    layer.feed_forward.w3.s = model_dict[key + ".feed_forward.w3.s"].to(DEVICE)
    layer.feed_forward.w3.shape = model_dict[key + ".feed_forward.w3.shape"]

model.to(device = DEVICE)

model_curr_dict = model.state_dict()
for k,v in list(model_curr_dict.items()):
    if (k.split(".")[-1] == "shape"):
        print(k, v)
    else:
        print(k, v.shape, v.dtype)

tokenizer = CodeLlamaTokenizer.from_pretrained("./CodeLlama-7b-Instruct-hf")

PROMPT = '''[INST] <<SYS>> You are a programmer, write the following python function that passes the given tests
<</SYS>>
Test Cases 
assert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3
assert max_chain_length([Pair(1, 2), Pair(3, 4),Pair(5, 6), Pair(7, 8)], 4) == 4
assert max_chain_length([Pair(19, 10), Pair(11, 12),Pair(13, 14), Pair(15, 16), Pair(31, 54)], 5) == 5

Write a function to find the longest chain which can be formed from the given set of pairs.
[/INST]
'''

model.eval()
input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
input_ids = input_ids.to(DEVICE)
print(input_ids.size())
print("Generating...")

print(tokenizer.batch_decode(input_ids, skip_special_tokens = True)[0])
start = time.time()
generated_ids = model.generate(input_ids, 1024, temperature=0.2, top_k=32, enc=tokenizer.batch_decode)
print("Tokens per second: ", (torch.prod(torch.tensor(list(generated_ids.size())))/(time.time() - start)).item())
print(generated_ids.size())
print(tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0])

