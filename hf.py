import torch
import os
import time

from transformers import LlamaForCausalLM, CodeLlamaTokenizer, GenerationConfig
import torch.nn.functional as F

DEVICE = "cuda:3"
DTYPE = torch.bfloat16
GROUP_SIZE = 64

tokenizer = CodeLlamaTokenizer.from_pretrained("./CodeLlama-7b-Instruct-hf")
model = LlamaForCausalLM.from_pretrained("./CodeLlama-7b-Instruct-hf")

PROMPT = '''[INST] <<SYS>> You are a programmer, write the following python function that passes the given tests
<</SYS>>
Test Cases 
assert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3
assert max_chain_length([Pair(1, 2), Pair(3, 4),Pair(5, 6), Pair(7, 8)], 4) == 4
assert max_chain_length([Pair(19, 10), Pair(11, 12),Pair(13, 14), Pair(15, 16), Pair(31, 54)], 5) == 5

Write a function to find the longest chain which can be formed from the given set of pairs.
[/INST]
'''

input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
input_ids = input_ids.to(DEVICE)
model.to(device = DEVICE, dtype = DTYPE)
print(input_ids)
print("Generating with config: ")
print(model.generation_config)

'''
generation_config = GenerationConfig(
    do_sample=True,
    temperature=1,
    top_p=1,
    top_k=32,
    num_beams=1,
    max_new_tokens=512
)
'''

start = time.time()
with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=1024)

print("Time taken", time.time()-start)
print(generated_ids)
filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
print(filling)

'''
temperature = 0.9
top_k = 32

with torch.no_grad():
    logits = model.forward(input_ids)

logits = logits.logits[:, -1, :] # crop to just the final time step

# pluck the logits at the final step and scale by desired temperature
logits = logits / temperature
# optionally crop the logits to only the top k options
v, top_k_idx = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')
print("top_k V: ", v)
print("top_k tokens:", top_k_idx)
# apply softmax to convert logits to (normalized) probabilities
probs = F.softmax(logits, dim=-1)
print("top_k prob:", torch.take(probs, top_k_idx))
idx_next = torch.multinomial(probs, num_samples=1)
print(idx_next)
'''

'''
======HF_Sample======
top_k tokens: tensor([[   13,  1753,  9294,  6113,  2220,  3057, 29961, 29937, 28956,  6778,
          9891, 18784, 29966, 29871,  1990, 14697,  1678, 29958,  6678,  1688,
          2277,  7692, 29874,  5215, 12008,  3539, 29896,  2457, 20547, 14153,
          1576,   268]], device='cuda:1')
top_k prob: tensor([[0.4972, 0.2483, 0.1527, 0.0331, 0.0083, 0.0067, 0.0063, 0.0051, 0.0044,
         0.0044, 0.0038, 0.0029, 0.0025, 0.0021, 0.0019, 0.0019, 0.0019, 0.0016,
         0.0014, 0.0014, 0.0014, 0.0013, 0.0011, 0.0010, 0.0010, 0.0010, 0.0010,
         0.0010, 0.0009, 0.0008, 0.0008, 0.0008]], device='cuda:1')
tensor([[13]], device='cuda:1')

======fp16======
top_k tokens: tensor([[   13,  1753,  9294,  6113,  2220,  3057, 29961, 29937,  6778, 28956,
          9891, 18784, 29966,  1678, 29871, 14697,  1990, 29958,  6678,  1688,
          2277, 29874,  7692,  3539, 12008, 29896,  5215,  2457, 20547, 14153,
          1576,   268]], device='cuda:1')
top_k prob: tensor([[0.4707, 0.2520, 0.1729, 0.0364, 0.0081, 0.0063, 0.0059, 0.0049, 0.0049,
         0.0043, 0.0034, 0.0030, 0.0025, 0.0020, 0.0020, 0.0020, 0.0018, 0.0016,
         0.0014, 0.0014, 0.0013, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011, 0.0010,
         0.0010, 0.0009, 0.0009, 0.0008, 0.0008]], device='cuda:1',
       dtype=torch.bfloat16)
tensor([[13]], device='cuda:1')

======hf======
top_k tokens: tensor([[29961, 28956,    13, 10605,  1753,   518,  4013,  1576, 29966,  2220,
         15945,  2266, 29871,  3166, 16159, 29912,  5215,  5113, 29898, 13696,
          9891, 13393,  1762, 29922,  3492,  9144, 29989, 29879,  3317,  8439,
           822, 29952]], device='cuda:0')
top_k prob: tensor([[9.9561e-01, 3.4682e-03, 3.9939e-04, 3.3283e-04, 1.0862e-04, 2.9541e-05,
         1.0792e-05, 6.9318e-06, 5.6281e-06, 5.2052e-06, 3.9427e-06, 2.6219e-06,
         2.0033e-06, 1.8209e-06, 1.4784e-06, 1.1900e-06, 1.0911e-06, 7.3187e-07,
         6.3146e-07, 5.2168e-07, 4.7008e-07, 4.2727e-07, 3.9174e-07, 3.6230e-07,
         3.5452e-07, 3.4094e-07, 3.3218e-07, 3.1396e-07, 3.0456e-07, 2.9545e-07,
         2.8290e-07, 2.7682e-07]], device='cuda:0')
tensor([[29961]], device='cuda:0')
'''