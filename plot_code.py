import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

data = np.load("./spec-mcts/stats/infer.npz")

results = torch.tensor(data["results"])
results_len = torch.tensor(data["results_len"])
batch_stats = torch.tensor(data["batch_stats"])

print(results.shape, results_len.shape, batch_stats.shape)
batch_sum = torch.zeros((128), dtype=torch.int64)
end_sum = torch.zeros((128), dtype=torch.int64)
active_sum = torch.zeros((128), dtype=torch.int64)

ROWS = 350
for i in range(0, ROWS):
    
    #print(results[0, 0, :])
    #print(results_len[i, :])

    ended = torch.sum((results_len[i, :]) <= (torch.arange(128).expand(100, -1).t()), axis=1)
    active_warps = F.relu(batch_stats[i, :] - ended)
    
    #print(ended, active_warps, batch_stats[i, :])
    #print(end_sum, active_sum, batch_sum)
    #breakpoint()
    end_sum += ended
    active_sum += active_warps
    batch_sum += batch_stats[i, :]

batch = batch_sum/ROWS
end = end_sum/ROWS
active = active_sum/ROWS

#print("Batch: ", batch)
#print("End: ", end)
#print("Active: ", active)

"""
fig, ax = plt.subplots()
ax.plot(ended, label="Batches Ended")
ax.plot(batch_stats[0, :], label="Running Batches")
ax.plot(active_warps, label = "Active Batches")

ax.set(xlabel='Time Step', ylabel='Batch Size', title='')
ax.grid()
fig.savefig("./spec-mcts/stats/BatchvTime.png")
"""

line1, = plt.plot(end, label="Batches Ended")
line2, = plt.plot(batch, label="Running Batches")
line3, = plt.plot(active, label = "Active Batches")

leg = plt.legend(loc='upper left')
plt.title("Batch Size vs Time Step")
plt.savefig("./spec-mcts/stats/BatchvTime.png")
