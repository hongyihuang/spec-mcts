import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
tps = np.load("./spec-mcts/stats/tps_tree.npy")

fig, ax = plt.subplots()
ax.plot(tps[0], tps[1])
ax.plot(tps[0], tps[2])

ax.set(xlabel='Batch Size', ylabel='Tokens per Second', title='')
ax.grid()

fig.savefig("./spec-mcts/stats/tps_tree.png")
