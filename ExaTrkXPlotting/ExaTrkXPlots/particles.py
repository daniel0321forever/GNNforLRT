#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots about particles in ExaTrkX routine.

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mc

from ExaTrkXPlotting import plot


@plot('exatrkx.particles.production_vertex', ['pairs', 'hits', 'particles'])
def production_vertices(ax, data):
    pairs = data['pairs']
    hits = data['hits']
    particles = data['particles']

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

    hits_with_particles = pd.merge(
        hits, particles,
        how='inner'
    )

    pairs_with_pid = pd.merge(
        pairs,
        hits_with_particles,
        left_on='hit_id_1',
        right_on='hit_id',
        how="inner"
    )

    # Group by vertex.
    pairs_group_by_vertices = pairs_with_pid.groupby(['vx', 'vy', 'vz'])

    # Create color map.
    colors = plt.cm.get_cmap('gnuplot', len(pairs_group_by_vertices) + 1)

    for idx, (vertex, pairs) in enumerate(pairs_group_by_vertices):
        vx, vy, vz = vertex

        # Get color.
        color = colors(idx)

        ax.scatter(
            vx, vy,
            marker='+',
            color=color,
            label=f'({vx}, {vy})'
        )


@plot('exatrkx.particles.types', ['pairs', 'hits', 'particles'])
def particle_types(ax, data):
    hits = data['hits']
    pairs = data['pairs']
    particles = data['particles']

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

    hits_with_particles = pd.merge(
        hits, particles,
        how='inner'
    )

    pairs_with_pid = pd.merge(
        pairs,
        hits_with_particles,
        left_on='hit_id_2',
        right_on='hit_id',
        how="inner"
    )

    # Group by vertex.
    pairs_group_by_vertices = pairs_with_pid.groupby(['vx', 'vy', 'vz'])

    for idx, (vertex, pairs) in enumerate(pairs_group_by_vertices):
        for particle_id, tracks in pairs.groupby('particle_id'):
            # particle_type = particle_types[int(tracks['particle_type'].iloc[0])]
            particle_type = int(tracks['particle_type'].iloc[0])

            if 'r_2' not in tracks.columns:
                tracks['r_2'] = np.sqrt(tracks['x']**2 + tracks['y']**2)

            x, y = tracks.loc[
                tracks['r_2'].idxmax()
            ][['x', 'y']]
            ax.annotate(particle_type, (x, y))


@plot('exatrkx.particles.tracks_with_production_vertex.2d', ['pairs', 'hits', 'particles'])
def particle_track_with_production_vertex(ax, data, line_width=0.1):
    """
    Plot hit pair 2D connections. Require hits dataframe and pairs dataframe.
    """
    hits = data['hits']
    pairs = data['pairs']
    particles = data['particles']

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

    hits_with_particles = pd.merge(
        hits, particles,
        how='inner'
    )

    pairs = pd.merge(
        pairs, hits_with_particles,
        left_on='hit_id_1',
        right_on='hit_id'
    )

    pairs = pd.merge(
        pairs, hits_with_particles,
        left_on='hit_id_2',
        right_on='hit_id',
        suffixes=('_1', '_2')
    )

    # Group by vertex.
    pairs_group_by_vertices = pairs.groupby(['vx_1', 'vy_1', 'vz_1'])

    # Create color map.
    colors = plt.cm.get_cmap('gnuplot', len(pairs_group_by_vertices) + 1)

    for idx, (vertex, pairs) in enumerate(pairs_group_by_vertices):
        # Get color.
        color = colors(idx)

        for particle_id, tracks in pairs.groupby('particle_id_1'):
            positions = pairs[['x_1', 'y_1', 'x_2', 'y_2']].to_numpy()
            position_pairs = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in positions]
            line_collection = mc.LineCollection(
                position_pairs,
                linewidths=line_width,
                color=color
            )
            ax.add_collection(line_collection)

        vx, vy, vz = vertex
        ax.scatter(
            vx, vy,
            marker='+',
            color=color,
            label=f'({vx}, {vy})'
        )

    ax.legend()
