#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# Data source.
from ExaTrkXDataIO import DataReader


def read_edges(reader: DataReader, columns: [str] = None):
    """
    Read edges from file.

    :param reader: Data reader.
    :param columns:
        Only keep columns you specified.
        Use for save memory.
    :return:
    """
    i = 0
    for event in reader.read():
        hits = event['hits'].join(
            event['particles'].set_index('particle_id'),
            on='particle_id'
        )

        # Filter columns to save memory.
        if columns is not None:
            hits = hits[columns]

        hit1 = hits.iloc[event['edges']['sender']].reset_index()
        hit2 = hits.iloc[event['edges']['receiver']].reset_index()

        df = hit1.join(
            hit2,
            lsuffix='_1',
            rsuffix='_2'
        ).assign(
            score=event['edges']['score'],
            truth=event['edges']['truth']
        )

        yield df


def edge_score_and_truth_label(edges, edge_filter=None):
    if edge_filter is not None:
        edges = edges[edge_filter]

    array = edges[['score', 'truth']].to_numpy()

    return array[:, 0], array[:, 1]


def read_hits(reader: DataReader):
    for event in reader.read():
        hits = event['hits'].join(
            event['particles'].set_index('particle_id'),
            on='particle_id'
        )
        yield hits
