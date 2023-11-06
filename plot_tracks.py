#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Plotter.
from ExaTrkXPlotting import Plotter, PlotConfig

# Include track plots.
import ExaTrkXPlots.tracks

if __name__ == '__main__':
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)

    # with pd.HDFStore('pileup_0/dataset/tracking/01', 'r') as fp:
        # df = fp['data']
    
    df = pd.read_csv('pileup_0/dataset/tracking/01.csv')
    generated = df
    reconstructable = df[df.is_reconstructable]
    matched = df[df.is_reconstructable & df.is_matched]

    Plotter(
        fig, {
            ax[0, 0]: PlotConfig(
                plot='exatrkx.tracks.distribution',
                args={
                    'var_col': 'pt',
                    'var_name': '$p_T$ [GeV]',
                    'bins': np.arange(0, 10.5, 0.5)
                }
            ),
            ax[0, 1]: PlotConfig(
                plot='exatrkx.tracks.efficiency',
                args={
                    'var_col': 'pt',
                    'var_name': '$p_T$ [GeV]',
                    'bins': np.arange(0, 10.5, 0.5)
                }
            ),
            ax[1, 0]: PlotConfig(
                plot='exatrkx.tracks.distribution',
                args={
                    'var_col': 'eta',
                    'var_name': r'$\eta$',
                    'bins': np.arange(-4.0, 4.1, 0.4)
                }
            ),
            ax[1, 1]: PlotConfig(
                plot='exatrkx.tracks.efficiency',
                args={
                    'var_col': 'eta',
                    'var_name': r'$\eta$',
                    'bins': np.arange(-4.0, 4.1, 0.4)
                }
            )
        },
        data={
            'generated': generated,
            'reconstructable': reconstructable,
            'matched': matched
        }
    ).plot(save="metrics/pileup_0/track_eff")

