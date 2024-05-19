#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots.tracks import (
    tracks,
    tracking_efficiency
)

from workflows import reconstruct_and_match_tracks
from plot_configurations import (
    particle_filters,
)


def plot_tracks(particles, save):
    fig, ax = plt.subplots(3, 2, figsize=(8, 8), tight_layout=True)
    plotter = Plotter(fig)

    # Setup data.
    generated = particles
    reconstructable = particles[particles.is_trackable]
    matched = particles[particles.is_trackable & particles.is_matched]

    print(f'#generated: {len(generated)}')
    print(f'#reconstructable: {len(reconstructable)}')
    print(f'#matched: {len(matched)}')

    plotter.data = {
        'generated': generated,
        'reconstructable': reconstructable,
        'matched': matched
    }

    plotter[ax[0, 0]] = PlotConfig(
        plot=tracks,
        args={
            'var_col': 'pt',
            'var_name': '$p_T$ [GeV]',
            'bins': 10 ** np.arange(0, 2.1, 0.1),
            'ax_opts': {
                'xscale': 'log'
            }
        }
    )

    plotter[ax[0, 1]] = PlotConfig(
        plot=tracking_efficiency,
        args={
            'var_col': 'pt',
            'var_name': '$p_T$ [GeV]',
            'bins': 10 ** np.arange(0, 2.1, 0.1),
            'ax_opts': {
                'xscale': 'log'
            }
        }
    )

    plotter[ax[1, 0]] = PlotConfig(
        plot=tracks,
        args={
            'var_col': 'eta',
            'var_name': r'$\eta$',
            'bins': np.arange(-4.0, 4.1, 0.4)
        }
    )

    plotter[ax[1, 1]] = PlotConfig(
        plot=tracking_efficiency,
        args={
            'var_col': 'eta',
            'var_name': r'$\eta$',
            'bins': np.arange(-4.0, 4.1, 0.4)
        }
    )

    plotter[ax[2, 0]] = PlotConfig(
        plot=tracking_efficiency,
        args={
            'var_col': 'd0',
            'var_name': '$d0$',
            'bins': 10 ** np.arange(0, 2.1, 0.1),
            'ax_opts': {
                'xscale': 'log'
            }
        }
    )

    plotter[ax[2, 0]] = PlotConfig(
        plot=tracking_efficiency,
        args={
            'var_col': 'd0',
            'var_name': '$d0$',
            'bins': 10 ** np.arange(0, 2.1, 0.1),
            'ax_opts': {
                'xscale': 'log'
            }
        }
    )

    plotter[ax[2, 1]] = PlotConfig(
        plot=tracking_efficiency,
        args={
            'var_col': 'z0',
            'var_name': '$z0$',
            'bins': 10 ** np.arange(0, 2.1, 0.1),
            'ax_opts': {
                'xscale': 'log'
            }
        }
    )

    plotter.plot(save=save)


if __name__ == '__main__':
    # root dir should be "analysis HSF"
    config_path = Path('configs')
    save = Path('../../output/tracks/')
    save.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(processes=8) as pool:

        reader = DataReader(
            # config_path='../configs/reading/processed/gnn.yaml',
            # config_path='../configs/reading/v4/processed/gnn.yaml',
            config_path=config_path/'gnn.yaml',
            base_dir="."
        )

        particles = pd.concat(
            pool.map(lambda x: reconstruct_and_match_tracks(
                data=x, epsilon=0.6), reader.read())
        )

    print(particles)

    # All.
    plot_tracks(
        particles,
        save=save / 'all.pdf'
    )

    # FIXME: The plot below requres parent type data, which is used to seperate displaced and prompt data
    # Displaced.
    # plot_tracks(
    #     particles[
    #         particle_filters['displaced'](particles)
    #     ],
    #     save=save / 'displaced.pdf'
    # )

    # # Prompt.
    # plot_tracks(
    #     particles[
    #         particle_filters['prompt'](particles)
    #     ],
    #     save=save / 'prompt.pdf'
    # )
