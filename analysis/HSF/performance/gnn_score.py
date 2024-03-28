#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import sklearn.metrics

# Plotting tools.
import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
import ExaTrkXPlots.performance

# Read edge, edge score and truth label.
from ExaTrkXDataIO import DataReader
from read_edges import read_edges, edge_score_and_truth_label


if __name__ == '__main__':
    gnn_arches = [
        "ResAGNN+ReLU",
        # "ResAGNN+SiLU",
        # "InteractionGNN+ReLU",
        # "InteractionGNN+SiLU",
        # "VanillaGCN+ReLU",
        # "VanillaGCN+SiLU"
    ]

    reader = DataReader(
        # config_path="../configs/reading/v4/processed/gnn.yaml",
        config_path="../configs/reading/processed/gnn.yaml",
        # config_path="../configs/reading/v4/processed/gnn_ttbar.yaml",
        base_dir="../../data"
    )

    for gnn_arch in gnn_arches:
        print(f'======{gnn_arch}======')

        # FIXME: Hacky way to change config. Maybe we can give this a proper API?
        reader.set_constant(
            'gnn_arch', gnn_arch
        )

        edges = pd.concat(list(read_edges(
            reader=reader,
            columns=['parent_ptype']
        )))

        edge_filters = {
            'All': None,
            'Displaced Tracks': (
                (edges['parent_ptype_1'].abs() == 50) |
                (edges['parent_ptype_2'].abs() == 50)
            ),
            'Prompt Tracks': (
                (edges['parent_ptype_1'].abs() == 24) |
                (edges['parent_ptype_2'].abs() == 24)
            )
        }

        for name, edge_filter in edge_filters.items():
            print(name)
            score, truth = edge_score_and_truth_label(edges, edge_filter)
            print(len(truth[truth > 0.5]))
            print(len(truth[truth <= 0.5]))
            print(len(score))

            # fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
            plt.rcParams.update({'font.size': 18})

            score, truth = edge_score_and_truth_label(edges, edge_filter)

            accuracy = sklearn.metrics.accuracy_score(truth, score > 0.5)
            precision = sklearn.metrics.precision_score(truth, score > 0.5)
            recall = sklearn.metrics.recall_score(truth, score > 0.5)
            print('Accuracy:            %.6f' % accuracy)
            print('Precision (purity):  %.6f' % precision)
            print('Recall (efficiency): %.6f' % recall)

            # Use precomputed values instead of re-calculate for each plot.
            false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(
                truth,
                score
            )
            precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
                truth,
                score
            )

            # Also compute AUC.
            auc = sklearn.metrics.auc(
                false_positive_rate,
                true_positive_rate
            )
            print('AUC:            %.6f' % auc)

            save = Path(f'../../output/plots/gnn/{gnn_arch}/4/')
            save.mkdir(parents=True, exist_ok=True)

            #fig.suptitle(name)

            Plotter(
                fig, {
                    # ax: PlotConfig(
                    #      plot='exatrkx.performance.score_distribution',
                    #      args={'title': name}
                    # )
                     ax: PlotConfig(
                         plot='exatrkx.performance.score_distribution',
                         args={'title': name}
                     ),
                     # ax[1]: PlotConfig(
                     #    plot='exatrkx.performance.roc_curve'
                     # ),
                     # ax[1, 0]: PlotConfig(
                     #     plot='exatrkx.performance.precision_recall_with_threshold'
                     # ),
                     # ax[1, 1]: PlotConfig(
                     #     plot='exatrkx.performance.precision_recall'
                     # )
                },
                data={
                    'truth': truth,
                    'score': score,
                    'false_positive_rate': false_positive_rate,
                    'true_positive_rate': true_positive_rate,
                    'precision': precision,
                    'recall': recall,
                    'thresholds': thresholds
                },
                # save=f'../../output/plots/gnn/{gnn_arch}/{name}.png'
                # save=f'../../output/plots/gnn/{gnn_arch}/{name}.pdf'
            ).plot(save=save/f'{name}.pdf')
