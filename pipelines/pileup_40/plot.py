import matplotlib.pyplot as plt
import yaml
import numpy as np

with open("tmp.yaml", 'r') as file:
    datas = yaml.load(file, yaml.FullLoader)

effs = [
    datas["emb_eff"],
    datas["fil_eff"],
    datas['gnn_eff'],
]

purs = [
    datas["emb_pur"],
    datas["fil_pur"],
    datas["gnn_pur"],
]

x = np.arange(3)

fig, ax = plt.subplots(nrows=2, figsize=(9, 12))
ax1, ax2 = ax
ax1.set_title("Stage Efficiency")
ax1.plot(effs)
ax1.set_ylabel("Efficieny")
ax1.set_xticks(x, ["embedding", "filtering", "GNN"])
for i, j in zip(x, effs):
    ax1.annotate(f"{j:.3f}", xy=(i, j))

ax2.set_title("Stage Purity")
ax2.plot(purs)
ax2.set_ylabel("Purity")
ax2.set_xticks(x, ["embedding", "filtering", "GNN"])
for i, j in zip(x, purs):
    ax2.annotate(f"{j:.3f}", xy=(i, j))

fig.savefig("stage_performance.png")