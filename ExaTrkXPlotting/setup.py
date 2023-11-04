#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='ExaTrkXPlotting',
    version='0.9.0',
    author='Ian Wang',
    url='https://github.com/rlf23240/ExaTrkXPlotting',
    description='Common plotting for ML routine.',
    install_requires=[
        'PyYAML',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn==0.11.1'
    ],
    packages=find_packages() + [
        'ExaTrkXPlots'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ]
)
