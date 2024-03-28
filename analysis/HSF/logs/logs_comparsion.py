#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

# Data source.
from ExaTrkXDataIO import DataReader

# Plotting tools.
import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots import hits, particles, pairs, train_logs


def plot_train_log(
    histories, train_log_name, val_log_name, save=None
):
    save = Path(save)

    save.mkdir(
        exist_ok=True,
        parents=True
    )

    # Request 1x2 figure for purity and efficiency plot.
    fig, ax = plt.subplots(
        1, 2, figsize=(16, 8), tight_layout=True
    )
    plotter = Plotter(fig)

    for label, history in histories.items():
        val_history = history[val_log_name]
        print(label)
        print('eff', val_history['efficiency'].tail(1))
        print('pur', val_history['purity'].tail(1))

        plotter[ax[0]].append(
            PlotConfig(
                plot=train_logs.train_log,
                args={
                    'tag': 'efficiency',
                    'label': label,
                    'y_label': 'Efficiency'
                },
                data={
                    'history': val_history
                }
            )
        )

        plotter[ax[1]].append(
            PlotConfig(
                plot=train_logs.train_log,
                args={
                    'tag': 'purity',
                    'label': label,
                    'y_label': 'Purity'
                },
                data={
                    'history': val_history
                }
            )
        )

    plotter.plot(save=save / 'pur_eff.pdf')

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 8), tight_layout=True
    )
    plotter = Plotter(
        fig, config='../configs/plotting/logs.yaml'
    )
    for label, history in histories.items():
        train_history = history[train_log_name]
        val_history = history[val_log_name]

        steps_per_epoch = int(
            len(train_history['loss']) / (len(val_history['loss']) - 1)
        )

        plotter[ax].append(
            PlotConfig(
                config='train_loss',
                data={
                    'history': train_history
                },
                args={
                    'steps_per_epoch': steps_per_epoch,
                    'label': f'{label} train loss'
                }
            )
        )

        plotter[ax].append(
            PlotConfig(
                config='val_loss',
                data={
                    'history': val_history
                },
                args={
                    'label': f'{label} validation loss'
                }
            )
        )

    plotter.plot(save=save / 'loss.pdf')


if __name__ == '__main__':
    # Draw one log only.
    model = Path('../../data/models/v5/small')
    output = Path('../../output/plots/v5/compare')
    cell = DataReader(
        config_path=model / 'cell/log.yaml',
        base_dir=model / 'cell'
    ).read_all()[0]

    nocell = DataReader(
        config_path=model / 'nocell/log.yaml',
        base_dir=model / 'nocell'
    ).read_all()[0]

    # Embedding.
    plot_train_log(
        histories={
            'With cell': cell,
            'Without cell': nocell,
        },
        train_log_name='embedding_train',
        val_log_name='embedding_val',
        save=output / 'embedding'
    )

    plot_train_log(
        histories={
            'With cell': cell,
            'Without cell': nocell,
        },
        train_log_name='filter_train',
        val_log_name='filter_val',
        save=output / 'filter'
    )

    """plot_train_log(
        histories={
            'With cell': cell,
            'Without cell': nocell,
        },
        train_log_name='gnn_train', 
        val_log_name='gnn_sig',
        save='../../output/plots/v2/geodigi/compare/gnn'
    )"""
