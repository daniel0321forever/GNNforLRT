from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

evtid = 1000

input = Path('../../data/trackml/train_100_events/')
output = Path('../../output/trackml')

# Read input data
cells = pd.read_csv(input / f'event{evtid:09}-cells.csv')
hits = pd.read_csv(input / f'event{evtid:09}-hits.csv')
particles = pd.read_csv(input / f'event{evtid:09}-particles.csv')
truth = pd.read_csv(input / f'event{evtid:09}-truth.csv')

print(cells)

particles['pt'] = np.sqrt(particles.px**2 + particles.py**2)
particles['eta'] = -np.log(np.tan(0.5*np.arctan2(
    particles.pt, particles.pz
)))

# Select volumes
# hits = hits[
#     # Endcap region
#     (hits.volume_id == 7) |
#     (hits.volume_id == 9) |
#     # Barrel region
#     (hits.volume_id == 8)
# ]
# Get truth infomation
hits = hits.merge(truth, on='hit_id', how='inner')
# Get particle information
hits = hits.merge(particles, on='particle_id', how='inner')
# Remove noise
hits = hits[hits.particle_id > 0]
print(hits)


# Compute transformation for detectors
detectors = pd.read_csv(input / f'detectors.csv')
matrices = np.zeros(shape=(len(detectors), 9))
for idx, (xu, xv, xw, yu, yv, yw, zu, zv, zw) in enumerate(detectors[[
    'rot_xu', 'rot_xv', 'rot_xw',
    'rot_yu', 'rot_yv', 'rot_yw',
    'rot_zu', 'rot_zv', 'rot_zw'
]].values):
    mat = np.linalg.inv(np.array([
        [xu, xv, xw],
        [yu, yv, yw],
        [zu, zv, zw]
    ]))
    matrices[idx, :] = mat.flatten()

detectors = pd.concat([detectors, pd.DataFrame(
    columns=[
        'rot_ux', 'rot_uy', 'rot_uz',
        'rot_vx', 'rot_vy', 'rot_vz',
        'rot_wx', 'rot_wy', 'rot_wz'
    ],
    data=matrices
)], axis=1)

cell_info = np.zeros(shape=(len(hits), 14))
for idx, (
    hid, volume_id, module_id, layer_id,
    # True momentum
    tpx, tpy, tpz,
    pt, eta
) in enumerate(tqdm(hits[[
    'hit_id', 
    'volume_id', 'module_id', 'layer_id', 
    'tpx', 'tpy', 'tpz',
    'pt', 'eta'
]].values)):
    # Collect cells associated to same hit
    hit_cells = cells[cells.hit_id == hid]
    cell_geometry = detectors[
        (detectors.volume_id == volume_id) &
        (detectors.module_id == module_id) &
        (detectors.layer_id == layer_id)
    ]

    ux, uy, uz, vx, vy, vz, wx, wy, wz = cell_geometry[[
        'rot_ux', 'rot_uy', 'rot_uz',
        'rot_vx', 'rot_vy', 'rot_vz',
        'rot_wx', 'rot_wy', 'rot_wz'
    ]].values[0, :]

    pitch_u, pitch_v = cell_geometry[[
        'pitch_u', 'pitch_v'
    ]].values[0, :]
    
    mat = np.array([
        [ux, vx, wx],
        [uy, vy, wy],
        [uz, vz, wz]
    ])

    tp = np.array([tpx, tpy, tpz])
    tpl = np.matmul(tp, mat)
    
    len_u = (max(hit_cells.ch0) - min(hit_cells.ch0) + 1) * pitch_u
    len_v = (max(hit_cells.ch1) - min(hit_cells.ch1) + 1) * pitch_v

    cell_info[idx] = [
        hid, volume_id,
        # Cell info
        len(hit_cells), hit_cells.value.sum(), len_u, len_v,
        # Local p
        tpl[0], tpl[1], tpl[2],
        # Global p
        tpx, tpy, tpz,
        # Particle information
        pt, eta
    ]

cell_info = pd.DataFrame(data=cell_info, columns=[
    'hit_id', 'volume_id',
    'cell_count', 'cell_val', 'len_u', 'len_v', 
    'tpu', 'tpv', 'tpw', 
    'tpx', 'tpy', 'tpz',
    'pt', 'eta',
])

output = Path('output/trackml')
output.mkdir(exist_ok=True, parents=True)
cell_info.to_csv(output/'cells.csv')

