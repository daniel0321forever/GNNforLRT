#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, AnyStr
from os import PathLike
from pathlib import Path
from time import time

import yaml

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .plot_config import PlotConfig
from .plot_manager import plot_manager


class Plotter:
    def __init__(
        self,
        fig: Figure,
        plots: Any = None,
        data: Any = None,
        config: Any = None,
    ):
        """
        Plotter of a figure.

        :param fig:
            matplotlib Figure object.
        :param plots:
            Configurations define how to plot each axes.
        :param config:
            Single or list of external configuration file path or config dict.
        :param data:
            Data pass to all plotting function if no data assign in configuration.
        """
        self.fig = fig
        self.plots = {axes: [] for axes in fig.get_axes()}

        if plots is not None:
            for ax, plot_config in plots.items():
                self.plots[ax] = plot_config

        self.config = config
        self.data = data

    def plot(
        self, save: Union[PathLike, AnyStr] = None, close: bool = True
    ):
        """
        Plot the figure.

        :param save:
            Figure save location. None if you want to show plot instead of save it.
        :param close:
            Whether close figure after plot complete to clean memory.
            This might be unwanted if you want to plot multiple time on same figure.
        """
        t_start = time()

        external_config = self._parse_external_configuration(self.config)

        for (ax, plt_config) in self.plots.items():
            if isinstance(plt_config, list):
                # If configuration is list
                # overlap different plot on same axes.
                for subplot_config in plt_config:
                    try:
                        self._plot(ax, subplot_config, external_config, self.data)
                    except RuntimeError as error:
                        print(error)
                        continue
            else:
                self._plot(ax, plt_config, external_config, self.data)

        if save is not None:
            save = Path(save)
            self.fig.savefig(save)

            print(
                f'Plot complete in {time()-t_start:.4f} second.\n'
                f'Figure output to {save.absolute()}\n'
            )
        else:
            plt.show()

        # Clean up.
        if close:
            plt.close(self.fig)

    def _parse_external_configuration(self, config) -> Dict[str, Any]:
        if isinstance(config, list):
            result = {}
            for sub_config in config:
                result = {
                    **result,
                    **(self._parse_external_configuration(sub_config))
                }

        if isinstance(config, str):
            config = Path(config)

        if isinstance(config, PathLike):
            with open(config) as fp:
                return yaml.load(fp, Loader=yaml.SafeLoader)

        if isinstance(config, dict):
            return config

        return {}

    def _plot(self, ax, plt_config, external_config, data):
        plt_type, plt_data, plt_args = plt_config.parse(
            external_config, data
        )

        if isinstance(plt_type, str):
            if plt_type in plot_manager.plots():
                print(f'Plotting {plt_type}...')
                plot_manager.plot(plt_type)(ax, plt_data, **plt_args)
            else:
                raise RuntimeError(f'Plot definition not found: {plt_type}. Skip.')
        else:
            print(f'Plotting {plt_type.name}...')
            plt_type(ax, plt_data, **plt_args)

    def __getitem__(self, ax):
        return self.plots[ax]

    def __setitem__(self, ax, value):
        self.plots[ax] = value
