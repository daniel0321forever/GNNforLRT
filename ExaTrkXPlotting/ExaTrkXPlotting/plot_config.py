#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict


class PlotConfig:
    """
    A plot configuration.
    """
    def __init__(
        self,
        plot: Any = None,
        data: Any = None,
        config: Any = None,
        args: Dict = None
    ):
        """
        Define a plot configuration.

        :param plot:
            Plot type ID or plotting object.
        :param data:
            Data pass to plotting function.
        :param config:
            External configuration.
            Either a string to reference external configuration or a config dictionary.
        :param args:
            Other kwargs pass to plotting function.
        """
        self.plot = plot
        self.data = data
        self.config = config
        self.kwargs = args or {}

    def parse(
        self,
        external_config,
        external_data
    ) -> [Any, Any, Dict[str, Any]]:
        config = self.config

        plot = self.plot
        data = self.data or external_data
        kwargs = self.kwargs

        if config is not None:
            if isinstance(config, str):
                if config in external_config:
                    config = external_config[config]

            if isinstance(config, dict):
                if 'plot' in config:
                    plot = plot or config['plot']
                if 'data' in config:
                    data = data or config['data']
                if 'args' in config:
                    kwargs = {
                        **self.kwargs,
                        **config['args']
                    }

        if plot is None:
            raise RuntimeError('Unrecognized plot configuration. Skip.')

        return plot, data, kwargs
