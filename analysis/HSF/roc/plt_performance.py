#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import os
from datetime import datetime
from matplotlib import pyplot as plt

# Plotter.
from ExaTrkXPlotting import Plotter, PlotConfig

# Include performance plots.
import ExaTrkXPlots.performance


def load_data(datas_dir: str):

    truths = []
    scores = []

    data_paths = os.listdir(datas_dir)
    print(data_paths)

    for data_path in data_paths:
        data_path = os.path.join(datas_dir, data_path)
        data = torch.load(data_path)
        truths.append(data['truth'])
        scores.append(data['score'])

    truths = torch.concat(truths, dim=-1).cpu().numpy()
    scores = torch.concat(scores, dim=-1).cpu().numpy()

    return truths, scores


if __name__ == '__main__':
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)

    # data = torch.load('/global/cfs/cdirs/m3443/usr/daniel/dataset/gnn/test/0125')
    # truth, score = data['truth'], data['score']
    # print("truth: ", truth.shape)
    # print("score: ", score.shape)

    # truth = truth.cpu().numpy()
    # score = score.cpu().numpy()

    edge_file_list = [
        "/global/cfs/cdirs/m3443/usr/daniel/dataset/gnn_1/test",
        "/global/cfs/cdirs/m3443/usr/daniel/dataset/gnn_2/test",
        "/global/cfs/cdirs/m3443/usr/daniel/dataset/gnn_3/test",
    ]

    for i in range(len(edge_file_list)):
        try:
            truths, scores = load_data(edge_file_list[i])
            # print(truths.shape)
            # print(scores.shape)

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
                fig=fig,
                plots={
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
                    'truth': truths,
                    'score': scores
                }
            ).plot(save=f"../../output/performance_{i}_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        except Exception as e:
            print(
                f"{e} occurs when loading data from {edge_file_list[i]}")

            continue
