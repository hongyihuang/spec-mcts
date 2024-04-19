import numpy as np
import torch
#import matplotlib.pyplot as plt

data = np.load("./spec-mcts/stats/passfail_t0.3.npz")
data2 = np.load("./spec-mcts/stats/infer_t0.3.npz")

success = torch.tensor(data["success"])
fail = torch.tensor(data["fail"])
print(success.shape, fail.shape)

ROWS = success.shape[0]

maxSuccess = torch.max(success, axis=1)[0]
print("Pass: ", torch.sum(maxSuccess >= 3))
print("Partial pass: ", torch.sum(maxSuccess >= 1))
print("Complete Fails: ", torch.sum(maxSuccess == 0))
print("Fails: ", torch.nonzero(maxSuccess == 0).flatten())

print("Pass ratio: ", torch.sum(maxSuccess >= 3)/ROWS)

print(maxSuccess)
print(maxSuccess.shape)

from datasets import load_dataset
from transformers import CodeLlamaTokenizer
tokenizer = CodeLlamaTokenizer.from_pretrained("./CodeLlama-7b-Instruct-hf")

dataset_full = load_dataset("mbpp")
print(dataset_full)

results = torch.tensor(data2["results"])
results_len = torch.tensor(data2["results_len"])
batch_stats = torch.tensor(data2["batch_stats"])

group = 'test'

desc = dataset_full[group]['text']
code = dataset_full[group]['code']
test = dataset_full[group]['test_list']
setup = dataset_full[group]['test_setup_code']
rows = dataset_full[group].num_rows
print(f"Found {group} {rows} rows. Iterating through solutions to check...")

i = 1
curr_len = results_len[i, 0]-6
print(results_len[i])
print(results[i, 0, :curr_len])
#breakpoint()
codestr = tokenizer.decode(results[i, 0, :curr_len], skip_special_tokens = True)
print(codestr)

breakpoint()
appended_code = setup[i] + "\r\n" + codestr + "\r\n" + test[i][0]
exec(appended_code, globals(), locals())
