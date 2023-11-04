#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class _PlotManager:
    def __init__(self):
        self._plots = {}

    def register(self, plot):
        self._plots[plot.name] = plot

    def plot(self, name):
        return self._plots.get(name, None)

    def plots(self):
        """
        Get current defined plots.

        :return: List of defined plots.
        """
        return list(self._plots.keys())


plot_manager = _PlotManager()
