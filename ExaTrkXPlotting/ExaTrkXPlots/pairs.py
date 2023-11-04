#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots about hit pairs (edges) in ExaTrkX routine.

For plot data requirement, detail list below:
    - hits:
        - required: hit_id, x, y, z or r, phi, z
    - pairs:
        - required: hit_id_1, hit_id_2
    - edges:
        - required: hit_id_1, hit_id_2,
        - optional: score
    - particles:
        - required: particle_id
        - optional: vx, vy, vz, parent_pid
    - truth:
        - required: hit_id, particle_id

For required columns, it use for all plot require this type of dataframe.
For optional columns, it use for special purpose and not required for all plots.
"""

from typing import Any

import numpy as np
import pandas as pd
from matplotlib import collections as mc

from ExaTrkXPlotting import plot


@plot('exatrkx.hit_pairs.2d', ['hits', 'pairs'])
def hit_pair_plot(ax, data, line_opts=None):
    """
    Plot hit pair 2D connections. Require hits dataframe and pairs dataframe.
    """
    hits = data['hits']
    pairs = data['pairs']

    if all(pd.Series(['x', 'y']).isin(hits.columns)):
        pass
    elif all(pd.Series(['r', 'phi']).isin(hits.columns)):
        # Cylindrical coord.
        r = hits['r']
        phi = hits['phi']

        # Compute cartesian coord
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        hits = hits.assign(
            x=x, y=y
        )
    else:
        raise KeyError('No valid coordinate data found.')

    pairs = pd.merge(
        pairs, hits,
        left_on='hit_id_1',
        right_on='hit_id'
    )

    pairs = pd.merge(
        pairs, hits,
        left_on='hit_id_2',
        right_on='hit_id',
        suffixes=('_1', '_2')
    )

    positions = pairs[['x_1', 'y_1', 'x_2', 'y_2']].to_numpy()
    position_pairs = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in positions]

    line_opts = {
        'linewidths': 0.1
    } | (line_opts or {})

    line_collection = mc.LineCollection(
        position_pairs, **line_opts
    )
    ax.add_collection(line_collection)

    ax.legend()


@plot('exatrkx.hit_pairs.hist', ['edges'])
def edge_hist(
    ax,
    data,
    feature: str,
    edge_filter=None,
    hist_opts: dict=None
):
    """
    Plot edge histogram. Require edges dataframe.
    Columns use by feature and edge_filter should also be exist in edges dataframe.
    """
    edges = data['edges']
    if edge_filter is not None:
        edges = edges[edge_filter]

    ax.set_xlabel(feature)

    hist_opts = {
        'lw': 2,
        'log': False,
        'density': False
    } | (his_opts or {})
    ax.hist(
        edges[feature], histtype='step', **hist_opts
    )

    ax.legend()
