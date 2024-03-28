#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from ExaTrkXDataIO import DataReader
from read_edges import read_hits

reader = DataReader(
    config_path="../configs/reading/processed/gnn.yaml",
    base_dir="../../data"
)
reader.variables['gnn_arch'] = ["ResAGNN+ReLU"]

n_hits = [0, 0, 0]
n_particles = [0, 0, 0]
for hits in read_hits(reader):
    n_hits[0] += len(hits)
    n_hits[1] += len(hits[
        hits['parent_ptype'].abs() == 50
    ])
    n_hits[2] += len(hits[
        hits['parent_ptype'].abs() == 24
    ])

    n_particles[0] += len(pd.unique(hits['particle_id']))
    n_particles[1] += len(pd.unique(hits[
        hits['parent_ptype'].abs() == 50
    ]['particle_id']))
    n_particles[2] += len(pd.unique(hits[
        hits['parent_ptype'].abs() == 24
    ]['particle_id']))

print(n_hits)
print(n_particles)
