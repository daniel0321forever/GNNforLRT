#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib import pyplot as plt

# Plotter.
from ExaTrkXPlotting import Plotter, PlotConfig

# Include hit and pair plots.
import ExaTrkXPlots.hits
import ExaTrkXPlots.pairs

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    Plotter(
        fig, {
            ax: PlotConfig(
                plot='exatrkx.hits.2d',
                data={
                    'hits': pd.read_csv(
                        'data/events/event000001000-hits.csv'
                    )
                }
            )
        }
    ).plot()
