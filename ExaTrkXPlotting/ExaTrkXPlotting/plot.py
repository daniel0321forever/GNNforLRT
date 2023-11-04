#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import functools

from .plot_manager import plot_manager


class Plot:
    """
    Plotting function wrapper.
    """
    def __init__(self, name, data_requirements: List, func):
        self.plot_func = func
        self.data_requirements = data_requirements
        self.name = name

        plot_manager.register(self)

    def __call__(self, ax, data, ax_opts=None, *args, **kwargs):
        if self.data_requirements is not None:
            # If data check is enable, check data with requirements.
            for requirement in self.data_requirements:
                if requirement not in data:
                    raise RuntimeError(
                        f'Data requirement for {self.name} not satisfy: {requirement}'
                    )

        self.plot_func(ax, data, *args, **kwargs)

        if ax_opts is not None:
            ax.set(**ax_opts)


def plot(name: str, data_requirements: List = None):
    """
    Decoration to define a plot.

    :param name:
        Type name of this plot. Use in all configurations plot_figure.
    :param data_requirements:
        Data requirements.
        This will be use to check input data before plotting.
        Only work if data is subscriptable.
        None if you want to disable this feature.
    :return:
        Decorator.
    """
    def decorator(func):
        plot = Plot(name, data_requirements, func)

        # Copy docstring and function signature.
        functools.update_wrapper(plot, func)

        return plot

    return decorator
