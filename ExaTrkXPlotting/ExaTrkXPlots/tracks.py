#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots about particle tracks in ExaTrkX routine.

For plot data requirement, detail list below:
    - generated:
        Generated tracks.

    - reconstructable:
        Reconstructable tracks.

    - matched:
        Matched tracks

No required column for those dataframes, but if you assign x_variable or track_filter,
then used column must exist.
"""

from typing import Callable
import math

import numpy as np

from ExaTrkXPlotting import plot


@plot('exatrkx.tracks.distribution', ['generated', 'reconstructable', 'matched'])
def tracks(
    ax,
    data,
    bins,
    var_col,
    var_name = None,
    track_filter: Callable = None,
    hist_opts: dict = None,
):
    """
    Plot track histogram.

    :param ax: matplotlib axis object.
    :param data: Data. Must contain column used in x_variable and track_filter.
    :param var_col: Column name to use as x axis of distribution. Must exist in data.
    :param var_name: Name to display as x axis label. Same as var_col if None.
    :param track_filter: Plot track pass filter only.
    :return:
    """
    generated = data['generated']
    reconstructable = data['reconstructable']
    matched = data['matched']

    # Apply filter.
    if track_filter is not None:
        generated = generated[track_filter(generated)]
        reconstructable = reconstructable[track_filter(reconstructable)]
        matched = matched[track_filter(matched)]

    hist_opts = {
        'lw': 2,
        'log': False
    } | (hist_opts or {})

    ax.hist(
        generated[var_col],
        label='Generated',
        histtype='step',
        bins=bins,
        **hist_opts
    )
    ax.hist(
        reconstructable[var_col],
        label='Reconstructable',
        histtype='step',
        bins=bins,
        **hist_opts
    )
    ax.hist(
        matched[var_col],
        label='Matched',
        histtype='step',
        bins=bins,
        **hist_opts
    )

    ax.set_ylabel('Events')
    ax.set_xlabel(var_name or var_col)

    ax.legend()
    ax.grid(True)


def _efficiency(matched, population):
    """
    Helper function to generate efficiency and statistical error for each bins.

    :param matched: Number of matched classification in each bins.
    :param population: Population in each bins.
    :return: Efficiency, Error
    """

    efficiency = [
        x / y if y != 0 else 0.0 for x, y in zip(
            matched, population
        )
    ]
    error = [
        math.sqrt(eff * (1.0 - eff) / y) if y != 0 else 0.0 for eff, y in zip(
            efficiency, population
        )
    ]

    return efficiency, error


@plot('exatrkx.tracks.efficiency', ['generated', 'reconstructable', 'matched'])
def tracking_efficiency(
    ax, data, bins, var_col, var_name=None, track_filter=None, errbar_opts=None
):
    """
    Plot track efficiency, both physical and technical, define as

    Physical Efficiency = #Matched / #Generated

    Technical Efficiency = #Matched / #Reconstructable

    :param ax: matplotlib axis object.
    :param data: Data. Must contain column used in x_variable and track_filter.
    :param var_col: Column name to use as x axis of distribution. Must exist in data.
    :param var_name: Name to display as x axis label. Same as var_col if None.
    :param bins: Bins of histogram.
    :param track_filter: Plot track pass filter only.
    :return:
    """

    generated = data['generated']
    reconstructable = data['reconstructable']
    matched = data['matched']

    # Apply filter.
    if track_filter is not None:
        generated = generated[track_filter]
        reconstructable = reconstructable[track_filter]
        matched = matched[track_filter]

    # Compute histogram.
    gen_hist, gen_bins = np.histogram(generated[var_col], bins=bins)
    reco_hist, reco_bins = np.histogram(reconstructable[var_col], bins=bins)
    matched_hist, matched_bins = np.histogram(matched[var_col], bins=bins)

    # Compute x location and error for each bin.
    xvals, xerrs = [], []
    for i in range(1, len(gen_bins)):
        xvals.append(0.5*(gen_bins[i]+gen_bins[i-1]))
        xerrs.append(0.5*(gen_bins[i]-gen_bins[i-1]))

    # Compute efficiency.
    physical_efficiency, physical_efficiency_error = _efficiency(
        matched_hist, gen_hist
    )
    technical_efficiency, technical_efficiency_error = _efficiency(
        matched_hist, reco_hist
    )

    # Plot physical and technical efficiency.
    ax.set_ylim(0.0, 1.05)

    errbar_opts = {
        'fmt': 'o',
        'lw': 2
    } | (errbar_opts or {})
    ax.errorbar(
        xvals, physical_efficiency,
        xerr=xerrs, yerr=physical_efficiency_error,
        label='Physical Efficiency',
        **errbar_opts
    )
    ax.errorbar(
        xvals, technical_efficiency,
        xerr=xerrs, yerr=technical_efficiency_error,
        label='Technical Efficiency',
        **errbar_opts
    )

    ax.set_xlabel(var_name or var_col)

    ax.legend()
    ax.grid(True)


@plot('exatrkx.tracks.efficiency.technical', ['generated', 'reconstructable', 'matched'])
def tracking_efficiency_techical(
    ax,
    data,
    bins,
    var_col,
    var_name: str = None,
    track_filter: Callable = None,
    errbar_opts: dict = None
):
    """
    Plot techical tracking efficiency, define as

    Technical Efficiency = #Matched / #Reconstructable

    :param ax: matplotlib axis object.
    :param data: Data. Must contain column used in x_variable and track_filter.
    :param var_col: Column name to use as x axis of distribution. Must exist in data.
    :param var_name: Name to display as x axis label. Same as var_col if None.
    :param bins: Bins of histogram.
    :param track_filter: Plot track pass filter only.
    :return:
    """

    generated = data['generated']
    reconstructable = data['reconstructable']
    matched = data['matched']

    # Apply filter.
    if track_filter is not None:
        generated = generated[track_filter(generated)]
        reconstructable = reconstructable[track_filter(reconstructable)]
        matched = matched[track_filter(matched)]

    # Compute histogram.
    gen_hist, gen_bins = np.histogram(generated[x_variable], bins=bins)
    reco_hist, reco_bins = np.histogram(reconstructable[x_variable], bins=bins)
    matched_hist, matched_bins = np.histogram(matched[x_variable], bins=bins)

    # Compute x location and error for each bin.
    xvals, xerrs = [], []
    for i in range(1, len(gen_bins)):
        xvals.append(0.5*(gen_bins[i]+gen_bins[i-1]))
        xerrs.append(0.5*(gen_bins[i]-gen_bins[i-1]))

    # Compute efficiency.
    technical_efficiency, technical_efficiency_error = _efficiency(
        matched_hist, reco_hist
    )

    ax.set_ylim(0.0, 1.05)

    # Plot technical efficiency.
    errbar_opts = {
        'fmt': 'o',
        'lw': 2
    } | (errbar_opts or {})
    ax.errorbar(
        xvals, technical_efficiency,
        xerr=xerrs, yerr=technical_efficiency_error,
        **errbar_opts
    )

    ax.set_xlabel(var_name or var_col)

    ax.legend()
    ax.grid(True)


@plot('exatrkx.tracks.efficiency.physical', ['generated', 'reconstructable', 'matched'])
def tracking_efficiency_physical(
    ax,
    data,
    bins,
    var_col,
    var_name=None,
    track_filter: Callable = None,
):
    """
    Plot physical tracking efficiency, define as

    Physical Efficiency = #Matched / #Generated

    :param ax: matplotlib axis object.
    :param data: Data. Must contain column used in x_variable and track_filter.
    :param bins: Bins of histogram.
    :param var_col: Column name to use as x axis of distribution. Must exist in data.
    :param var_name: Name to display as x axis label. Same as var_col if None.
    :param track_filter: Plot track pass filter only.
    :return:
    """

    generated = data['generated']
    reconstructable = data['reconstructable']
    matched = data['matched']

    # Apply filter.
    if track_filter is not None:
        generated = generated[track_filter(generated)]
        reconstructable = reconstructable[track_filter(reconstructable)]
        matched = matched[track_filter(matched)]

    # Compute histogram.
    gen_hist, gen_bins = np.histogram(generated[var_col], bins=bins)
    reco_hist, reco_bins = np.histogram(reconstructable[var_col], bins=bins)
    matched_hist, matched_bins = np.histogram(matched[var_col], bins=bins)

    # Compute x location and error for each bin.
    xvals, xerrs = [], []
    for i in range(1, len(gen_bins)):
        xvals.append(0.5*(gen_bins[i]+gen_bins[i-1]))
        xerrs.append(0.5*(gen_bins[i]-gen_bins[i-1]))

    # Compute efficiency.
    physical_efficiency, physical_efficiency_error = _efficiency(
        matched_hist, gen_hist
    )

    ax.set_ylim(0.0, 1.05)

    # Plot physical efficiency.
    errbar_opts = {
        'fmt': 'o',
        'lw': 2
    } | (errbar_opts or {})
    ax.errorbar(
        xvals, physical_efficiency,
        xerr=xerrs, yerr=physical_efficiency_error,
        **errbar_opts
    )

    ax.set_xlabel(var_name or var_col)

    ax.legend()
    ax.grid(True)

