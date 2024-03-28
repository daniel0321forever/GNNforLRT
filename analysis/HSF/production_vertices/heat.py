import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

evtids = [1]

all_particles = []

nhit = 0
nparticles = 0

for evtid in evtids:
    npz_data = np.load(f'data/v2/raw/{evtid}.npz')
    particles = pd.read_csv(f'data/v2/particles/event{evtid:09}-particles.csv')
    x = npz_data['x']

    pid = pd.DataFrame(
        data={'pid': npz_data['pid']}
    )
    pid = pid.merge(particles, left_on='pid', right_on='particle_id', how='inner')
    pid = pid.fillna(0)

    all_particles.append(particles)
    hits = npz_data['x']

all_particles = pd.concat(all_particles)

fig, ax = plt.subplots(1, 1, figsize=(12, 4), tight_layout=True)

ax.set_title('Production Vertex Position')

h = ax.hist2d(
    x=all_particles.vz,
    y=np.sqrt(all_particles['vx']**2 + all_particles['vy']**2),
    bins=(
        np.arange(-10, 10, 0.1),
        np.arange(0, 10, 0.1)
        # np.arange(-500, 500, 1),
        # np.arange(0, 500, 1)
    ),
    # cmap='Reds',
    norm=LogNorm()
)
ax.set_xlabel('z [mm]')
ax.set_ylabel('r [mm]')

fig.colorbar(h[3], ax=ax, label='#Particles')
# ax.scatter(hits[:, 2], np.sqrt(hits[:, 0]**2 + hits[:, 1]**2), s=0.01)

fig.savefig('output/plots/v5/v2.pdf')
