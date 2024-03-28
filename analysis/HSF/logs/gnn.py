#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Data source.
from ExaTrkXDataIO import DataReader

from logs import plot_train_log


if __name__ == '__main__':
    for log in DataReader(
        config_path='../configs/reading/logs/gnn.yaml',
        base_dir='../../data'
    ).read():
        print(f"======{log.gnn_arch}======")

        plot_train_log(
            log['gnn_train'],
            log['gnn_val'],
            f'../../output/plots/gnn/{log.gnn_arch}'
        )
