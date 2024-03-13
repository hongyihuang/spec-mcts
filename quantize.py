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

DEVICE = "mps"
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
    scale = wmax / 127.0
    scale = scale.type(DTYPE)
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)

    return int8val, scale #, maxerr

def dequantize_q80(w, scale, group_size, shape, ptdtype=torch.float32):
    """
    takes a Q8_0 tensor and returns the float32 version
    """
    w = w.view(-1, group_size)

    # dequantize by rescaling
    fpval = (w.type(ptdtype) * scale[:,None]).view(-1)
    return fpval.reshape(shape)

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
    # scale into range [-7, 7]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    MSB = int8val.reshape(-1, 2, group_size)[:, 0, :]
    LSB = int8val.reshape(-1, 2, group_size)[:, 1, :]
    int8val = (MSB << 4) | (LSB & 0x0F)

    return int8val, scale #, maxerr

def dequantize_q40(w, scale, group_size, shape, ptdtype):
    """
    takes a Q4_0 tensor and returns the dequantized version
    """
    w = w.view(-1, group_size)
    MSB = (w & 0xF0) >> 4
    LSB = w & 0x0F
    w = torch.hstack((MSB, LSB)).view(-1, group_size)

    # dequantize by rescaling
    fpval = (w.type(ptdtype) * scale[:,None]).view(-1)
    return fpval.reshape(shape)

'''
def sample(prompt="Once upon a time, ", max_new_tokens=128, temperature=0.9, top_k=32, num_samples=1, enc = None):
    vocab_size = ModelArgs.vocab_size

    if enc is None:
        prompt_ids = enc.encode(prompt, bos=True, eos=False)
    idx = (torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                for _ in range(max_new_tokens):
                    # if the sequence context is growing too long we must crop it at block_size
                    idx_cond = idx if idx.size(1) <= ModelArgs.max_seq_len else idx[:, -ModelArgs.params.max_seq_len:]
                    # forward the model to get the logits for the index in the sequence
                    logits = model(idx_cond) # dummy Y, doesn't matter
                    logits = logits[:, -1, :] # crop to just the final time step
                    if temperature == 0.0:
                        # "sample" the single most likely index
                        _, idx_next = torch.topk(logits, k=1, dim=-1)
                    else:
                        # pluck the logits at the final step and scale by desired temperature
                        logits = logits / temperature
                        # optionally crop the logits to only the top k options
                        if top_k is not None:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float('Inf')
                        # apply softmax to convert logits to (normalized) probabilities
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                    # append sampled index to the running sequence and continue
                    y = torch.cat((idx, idx_next), dim=1)
                print(enc.decode(y[0].tolist()))
                print(y)
                print('---------------')
'''

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

class LinearQ4_0(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear_w, self.linear_s = quantize_q80(linear.weight, GROUP_SIZE)
        self.linear_w.to(DEVICE)
        self.linear_s.to(DEVICE)
        self.linear_shape = linear.weight.shape
        del linear

    def forward(self, x):
        #start = time.time()
        deq = dequantize_q80(self.linear_w, self.linear_s, GROUP_SIZE, self.linear_shape, DTYPE).to(DEVICE)
        #print("Dequantize time: ", time.time() - start)
        #start = time.time()
        result = F.linear(x, deq)
        #print("Linear time: ", time.time() - start)
        return result

model = Transformer(ModelArgs) #default is llama7B
model.load_state_dict(model_dict, strict=False, assign=True)

model.eval()
assign_lora = partial(LinearQ4_0)

for i, layer in enumerate(model.layers):
    layer.attention.wq = LinearQ4_0(layer.attention.wq)
    layer.attention.wk = LinearQ4_0(layer.attention.wk)
    layer.attention.wv = LinearQ4_0(layer.attention.wv)
    layer.attention.wo = LinearQ4_0(layer.attention.wo)
    layer.feed_forward.w1 = LinearQ4_0(layer.feed_forward.w1)
    layer.feed_forward.w2 = LinearQ4_0(layer.feed_forward.w2)
    layer.feed_forward.w3 = LinearQ4_0(layer.feed_forward.w3)

    layer.attention_norm.to(device = DEVICE, dtype = DTYPE)
    layer.ffn_norm.to(device = DEVICE, dtype = DTYPE)

model.output.to(device = DEVICE, dtype = DTYPE) # = LinearQ4_0(model.output)
model.norm.to(device = DEVICE, dtype = DTYPE)
#print(model)
model.to(device = DEVICE, dtype = DTYPE)

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

