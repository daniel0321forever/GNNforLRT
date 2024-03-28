#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots import hits, pairs, particles

if __name__ == '__main__':
    reader = DataReader(
        config_path='../configs/reading/hits.yaml',
        base_dir='../../data'
    )

    event_ids = range(455, 458)

    plt.rcParams.update({'font.size': 16})

    for event_id in event_ids:
        data = reader.read_one(evtid=event_id)

        save = Path(f'../../output/plots/10k+5k/overview/event{data.evtid}')
        save.mkdir(exist_ok=True, parents=True)

        hit_data = data['hits']
        particle_data = data['particles']
        pair_data = pd.DataFrame(data={
            'hit_id_1': hit_data.iloc[data['edges']['sender']]['hit_id'].to_numpy(),
            'hit_id_2': hit_data.iloc[data['edges']['receiver']]['hit_id'].to_numpy()
        })

        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

        Plotter(fig, {
            ax: [
                PlotConfig(
                    plot=hits.hit_plot,
                    data={
                        'hits': hit_data
                    }
                ),
                #PlotConfig(
                #    plot=pairs.hit_pair_plot,
                #    data={
                #        'hits': hit_data,
                #        'pairs': pair_data
                #    }
                #)
            ]
        }).plot(save=save/'hits.pdf')

        hit_with_particles = pd.merge(
            hit_data,
            particle_data[['particle_id', 'particle_type']],
            on='particle_id'
        )
        hit_filtered = hit_with_particles[hit_with_particles['particle_type'].isin([13, -13])]
        pair_filtered = pair_data[
            pair_data['hit_id_1'].isin(hit_data['hit_id']) &
            pair_data['hit_id_2'].isin(hit_data['hit_id'])
        ]

        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

        Plotter(fig, {
            ax: [
                PlotConfig(
                    plot=hits.hit_plot,
                    data={
                        'hits': hit_data
                    }
                ),
                PlotConfig(
                    plot=particles.particle_track_with_production_vertex,
                    data={
                        'hits': hit_filtered,
                        'pairs': pair_filtered,
                        'particles': particle_data,
                        'truth': data['truth']
                    },
                    args={
                        'line_width': 1.0
                    }
                ),
                PlotConfig(
                    plot=particles.particle_types,
                    data={
                        'hits': hit_filtered,
                        'pairs': pair_filtered,
                        'particles': particle_data,
                        'truth': data['truth']
                    }
                )

            ]
        }).plot(save=save/'particles.pdf')

