#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from matplotlib import pyplot as plt

# Plotter.
from ExaTrkXPlotting import Plotter, PlotConfig

# Include performance plots.
import ExaTrkXPlots.performance

if __name__ == '__main__':
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)

    data = torch.load('pileup_0/dataset/GNN/test/0003')
    truth, score = data['truth'], data['score']

    truth = truth.cpu().numpy()
    score = score.cpu().numpy()
    print(type(truth), type(score))
    # You can also precompute values and pass to plotter in data
    # to avoid multiple computation in each plot if many plots share same data.
    """
    import sklearn.metrics
    
    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(
        truth, 
        score
    )
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
        truth,
        score
    )
    """

    Plotter(
        fig, {
            ax[0, 0]: PlotConfig(
                plot='exatrkx.performance.score_distribution'
            ),
            ax[0, 1]: PlotConfig(
                plot='exatrkx.performance.roc_curve'
            ),
            ax[1, 0]: PlotConfig(
                plot='exatrkx.performance.precision_recall_with_threshold'
            ),
            ax[1, 1]: PlotConfig(
                plot='exatrkx.performance.precision_recall'
            )
        },
        data={
            'truth': truth,
            'score': score
        }
    ).plot(save="metrics/pileup_0/performance")
