import torch
from hf_model_q40_kv_shared import Transformer, ModelArgs
from transformers import CodeLlamaTokenizer
import torch.nn.functional as F
import numpy as np
from mcts import mcts

from tqdm import tqdm
from datasets import load_dataset

dataset_full = load_dataset("mbpp")
print(dataset_full)

group = 'test'

desc = dataset_full[group]['text']
code = dataset_full[group]['code']
test = dataset_full[group]['test_list']
setup = dataset_full[group]['test_setup_code']
rows = dataset_full[group].num_rows
print(f"Found {group} {rows} rows. Iterating through solutions to check...")

success_count = 0
fail_count = 0

for i in tqdm(range(rows)):
    success = True
    for j in range(len(test[i])):
        try:
            appended_code = code[i] + "\r\n" + setup[i] + "\r\n" + test[i][j]
            
            exec(appended_code, globals(), locals())
            success_count += 1
        except Exception as e:
            print("Test failed: ", i, j, test[i][j], e)
            fail_count += 1
            success = False
    if not success:
        print("Code: ", code[i])

print("Success: ", success_count)
print("Fail: ", fail_count)

PROMPT = '''[INST] <<SYS>> You are a programmer, write a python function that passes the given tests. <</SYS>>
Function task: 
{desc[0]}

Given test cases: 
{"\n".join(test[0])}
[/INST]
[PYTHON]
{setup[0]}
'''

#print("Example prompt: \n", PROMPT)
#print("Example solution: \n", code[0] + "\r\n" + setup[0] + "\r\n" + "\n".join(test[0]))

DEVICE = "cuda"
DTYPE = torch.float16
GROUP_SIZE = 64
TEMP = 0.7
BATCH_SIZE = 1
SEQ_LEN = 256
PROMPT_LEN = 512

print(torch.__version__)

def memoryStats():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)

    print("Showing Device 0 Memory Stats")
    print("Total Memory (GB): ", t/(1024**3))
    print("Reserved Memory (GB): ", r/(1024**3))
    print("Allocated Memory (GB): ", a/(1024**3))

memoryStats()

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

model_dict = {}
print("Loading file: ", model_file)
curr_dict = remap_names(model_file)
model_dict.update(curr_dict)

'''
for k,v in list(model_dict.items()):
    if (k.split(".")[-1] == "shape"):
        print(k, v)
    else:
        print(k, v.shape, v.dtype)
'''

modelArgs = ModelArgs(max_batch_size = BATCH_SIZE, 
                      max_prompt_seq_len = PROMPT_LEN,
                      max_seq_len = SEQ_LEN)
model = Transformer(modelArgs) #default is llama7B
model.load_state_dict(model_dict, strict=False, assign=True)
memoryStats()

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

model_dict = {} # deallocate CPU memory for model
model.to(device = DEVICE, dtype = DTYPE)
#model = torch.compile(model)
torch.backends.cuda.enable_flash_sdp(enabled = True)
torch.cuda.device(DEVICE)

'''
model_curr_dict = model.state_dict()
for k,v in list(model_curr_dict.items()):
    if (k.split(".")[-1] == "shape"):
        print(k, v)
    else:
        print(k, v.shape, v.dtype)
'''

tokenizer = CodeLlamaTokenizer.from_pretrained("./CodeLlama-7b-Instruct-hf")
tree = mcts(model, depth = SEQ_LEN, nodes=0, top_k=32, temp=TEMP)

print("NOW RUNNING LLAMA 7B CODER!")

success_count = 0
fail_count = 0

model.eval()

all_results = torch.zeros((rows, BATCH_SIZE, SEQ_LEN), dtype=torch.int16)
all_results_len = torch.zeros((rows, BATCH_SIZE), dtype=torch.int16)
all_batch_stats = torch.zeros((rows, SEQ_LEN), dtype=torch.int16)

# Test saving empty to prevent directory error after running...
SAVE = True
SAVE_PATH = f"./spec-mcts/stats/infer_t{TEMP}_q{SEQ_LEN}_b{BATCH_SIZE}"

for i in tqdm(range(rows), desc="Tasks"):
    print(f"Running row {i}")
    success = True
    PROMPT = f'''[INST] <<SYS>> You are a programmer, write a python function that passes the given tests. <</SYS>>
Function task: 
{desc[i]}

Given test cases: 
{"\n".join(test[i])}
[/INST]
[PYTHON]
{setup[i]}'''
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(DEVICE)
    if (input_ids.numel() > PROMPT_LEN):
        print(f"WARNING: PROMPT LEN {input_ids.size()} > {PROMPT_LEN} in row {i}")
        continue

    results, results_len, batch_stats = tree.search(input_ids, BATCH_SIZE)
    all_results[i, :results.shape[0], :results.shape[1]] = results
    all_results_len[i, :results_len.shape[0]] = results_len
    all_batch_stats[i, :batch_stats.shape[0]] = batch_stats
    #breakpoint()
    print("Length", results_len[0])
    print(tokenizer.batch_decode(results, skip_special_tokens = True)[0])
    #print(results[0, :results_len[0]])
    if (i % 50 == 0) & SAVE:
        np.savez(SAVE_PATH, 
                 results = all_results, 
                 results_len = all_results_len, 
                 batch_stats = all_batch_stats)

np.savez(SAVE_PATH, results = all_results, results_len = all_results_len, batch_stats = all_batch_stats)
