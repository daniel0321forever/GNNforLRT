#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots about train history.

Plots about particles in ExaTrkX routine.

For plot data requirement, detail list below:
    - history:
        Train history. Either array or dict.
        For dict, dict must contain data assign by tag.
        For array, treated as value for each step.
"""

import numpy as np
from matplotlib.ticker import MaxNLocator

from ExaTrkXPlotting import plot


@plot('exatrkx.train_log', ['history'])
def train_log(ax, data, tag=None, steps_per_epoch=1, plot_opts=None):
    history = data['history']
    if tag is None:
        values = history
    else:
        values = history[tag]

    train_epochs = np.arange(0, len(values)) * 1.0/steps_per_epoch

    plot_opts = plot_opts or {}
    ax.plot(train_epochs, values, **plot_opts)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel(tag or 'History')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend()

