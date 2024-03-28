#!/usr/bin/env python3
#-*- coding: utf-8 -*-


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output = Path('../../output/plots/trackml/cells')
output.mkdir(parents=True, exist_ok=True)

cells = pd.read_csv('../../output/trackml/cells.csv')

# 1 GeV cut
# cells = cells[cells.pt >= 1.0]
cells = cells[cells.volume_id.isin([
    7, 8, 9
])]

theta = np.arctan2(cells.tpv, cells.tpw)
phi = np.arctan2(cells.tpu, cells.tpw)
eta = cells.eta

for name, label, variable in [
    ('t', r'$\theta$', theta),
    ('p', r'$\phi$', phi),
    ('e', r'$\eta$', eta)
]:
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    ax = axes[0, 0]
    ax.scatter(variable, cells.len_u)
    ax.set_ylabel('Cluster Height(u) [mm]')
    ax.set_xlabel(label)

    ax = axes[0, 1]
    ax.scatter(variable, cells.len_v)
    ax.set_ylabel('Cluster Width(v) [mm]')
    ax.set_xlabel(label)

    ax = axes[1, 0]
    ax.scatter(variable, cells.cell_count)
    ax.set_ylabel('Cell Count')
    ax.set_xlabel(label)

    ax = axes[1, 1]
    ax.scatter(variable, cells.cell_val)
    ax.set_ylabel('Charge Disposed')
    ax.set_xlabel(label)

    fig.savefig(output / f'{name}.png')

