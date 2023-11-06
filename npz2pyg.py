#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

parser = ArgumentParser()

parser.add_argument(
    '--input', '-i',
    type=Path,
    required=True,
    help='Input directory containing npz file'
)

parser.add_argument(
    '--output', '-o',
    type=Path,
    required=True,
    help='Output directory for pyg2 file'
)

args = parser.parse_args()

source = args.input

target = args.output
target.mkdir(parents=True, exist_ok=True)

for source_file in tqdm(list(source.glob('*'))):
    # Init a new PyG data.
    pyg_data = Data()
    
    name_idx = source_file.name.split(".")[0]
    evtid = int(name_idx)
    npz_data = np.load(source_file)

    # print([key for key in npz_data.keys()])
    # print(npz_data['particles'].shape)
    # Construct particle information
    particles = pd.DataFrame(
        columns=['pid', 'particle_type', 'process', 'vx', 'vy', 'vz', 'vt', 'px', 'py', 'pz', 'm', 'q', 'parent_pid', 'pt', 'radius', 'eta'],
        data=npz_data['particles']
    ).drop_duplicates(
        subset=['pid']
    )
    pid = pd.DataFrame(
        data={'pid': npz_data['pid']}
    )
    pid = pid.merge(particles, on='pid', how='left')
    pid = pid.fillna(0)

    # Copy data into PyG data
    pyg_data.hid = torch.from_numpy(npz_data['hid']).long()
    pyg_data.pid = torch.from_numpy(npz_data['pid']).long()
    pyg_data.layerless_true_edges = torch.from_numpy(npz_data['layerless_true_edges']).long()
    pyg_data.layers = torch.from_numpy(npz_data['layers']).float()
    pyg_data.pt = torch.from_numpy(pid.pt.values).float()
    pyg_data.eta = torch.from_numpy(pid.eta.values).float()

    # Already scaled!
    pyg_data.x = torch.from_numpy(npz_data['x']).float()

    # Do we need full path?
    pyg_data.event_file = f'{evtid:04}'

    torch.save(pyg_data, target / f'{evtid:04}')

