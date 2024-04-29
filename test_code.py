import numpy as np
import torch
#import matplotlib.pyplot as plt
from transformers import CodeLlamaTokenizer
import tqdm
from datasets import load_dataset
import signal
import time

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

# load data
print("Loading data")
PATH = "t0.7_q256_b100"
data = np.load("./spec-mcts/stats/infer_"+PATH+".npz")
dataset_full = load_dataset("mbpp")
print(dataset_full)

results_x0fsa = torch.tensor(data["results"])
results_len_x0fsa = torch.tensor(data["results_len"])
batch_stats_x0fsa = torch.tensor(data["batch_stats"])

group = 'test'

desc_x0fsa = dataset_full[group]['text']
code_x0fsa = dataset_full[group]['code']
test_x0fsa = dataset_full[group]['test_list']
setup_x0fsa = dataset_full[group]['test_setup_code']
rows_x0fsa = dataset_full[group].num_rows
print(f"Found {group} {rows_x0fsa} rows. Iterating through solutions to check...")

ROWS = 500
SEQ_LEN = results_x0fsa.shape[2]
BATCH_SIZE = results_x0fsa.shape[1]

print(results_x0fsa.shape, results_len_x0fsa.shape, batch_stats_x0fsa.shape)

success_count_x0fsa = 0
fail_count_x0fsa = 0
for i in tqdm.tqdm(range(ROWS)):
    success = True
    for j in range(len(test_x0fsa[i])):
        try:
            appended_code = setup_x0fsa[i] + "\r\n" + code_x0fsa[i] + "\r\n" + test_x0fsa[i][j]
            exec(appended_code, globals(), locals())
            success_count_x0fsa += 1
        except Exception as e:
            print("Test failed: ", i, j, test_x0fsa[i][j], e)
            fail_count_x0fsa += 1
            success = False
    if not success:
        print("Code: ", code_x0fsa[i])

print("Success: ", success_count_x0fsa)
print("Fail: ", fail_count_x0fsa)

print("Now iterating over generated code...")
success = torch.zeros((ROWS, BATCH_SIZE), dtype=torch.uint8)
fail = torch.zeros((ROWS, BATCH_SIZE), dtype=torch.uint8)
success_count = 0
fail_count = 0

tokenizer = CodeLlamaTokenizer.from_pretrained("./CodeLlama-7b-Instruct-hf")

overflows = 0
for i in range(ROWS):
    input_ids = tokenizer(code_x0fsa[i], return_tensors="pt")["input_ids"]
    if (input_ids.numel() > SEQ_LEN):
        print(f'WARNING: Solution on row {i} has {input_ids.numel()} > {SEQ_LEN} tokens')
        overflows += 1

print("Total tasks out of sequence range: ", overflows)

bar = tqdm.tqdm(range(ROWS))

for i_x0fsa in bar:
    passrate = round(success_count/(success_count+fail_count+0.001), 4)
    bar.set_description_str(f'Pass Rate {passrate}')
    bar.refresh()
    rowSuccess_x0fsa = False
    for b_x0fsa in range(max(batch_stats_x0fsa[i_x0fsa])):
        # -6 for [/PYTHON]
        isSuccess_x0fsa = True
        codestr = tokenizer.decode(results_x0fsa[i_x0fsa, b_x0fsa, :results_len_x0fsa[i_x0fsa, b_x0fsa]-6], skip_special_tokens = True)
        #breakpoint()
        for j_x0fsa in range(len(test_x0fsa[i_x0fsa])):
            try:
                appended_code = setup_x0fsa[i_x0fsa] + "\r\n" + codestr + "\r\n" + test_x0fsa[i_x0fsa][j_x0fsa]    
                with timeout(seconds=1):        
                    exec(appended_code, globals(), locals())
                success[i_x0fsa, b_x0fsa] += 1
            except Exception as e:
                fail[i_x0fsa, b_x0fsa] += 1
                #print(f"Test failed: row {i} batch {b} {codestr} error: {e}")
                isSuccess_x0fsa = False
        rowSuccess_x0fsa |= isSuccess_x0fsa
        #if rowSuccess_x0fsa:
        #    break # comment out since we want to know pass rate at less than that batch
    if rowSuccess_x0fsa:
        success_count += 1
    else:
        fail_count += 1
    #print(success[i], fail[i])

#print(success)
#print(fail)
maxSuccess = torch.max(success, axis=1)[0]
print(torch.sum(maxSuccess >= 3))
print("Pass ratio: ", torch.sum(maxSuccess >= 3)/ROWS)

np.savez("./spec-mcts/stats/passfail_"+PATH, success=success, fail=fail)

"""
line1, = plt.plot(end, label="Batches Ended")
line2, = plt.plot(batch, label="Running Batches")
line3, = plt.plot(active, label = "Active Batches")

leg = plt.legend(loc='upper left')
plt.title("Batch Size vs Time Step")
plt.savefig("./spec-mcts/stats/BatchvTime.png")
"""
