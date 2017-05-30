"""
Pyret
=====

A Python package for analysis of retinal data

Modules
-------
spiketools      - Tools for manipulating spike trains
nonlinearities  - Methods for estimating nonlinearities
stimulustools   - Tools for getting stimulus history and statistics
visualizations  - Tools for plotting receptive fields, filters, etc.
filtertools     - Analysis of linear filters

For more information, see the accompanying README.md
"""
__all__ = [
    'spiketools',
    'nonlinearities',
    'stimulustools',
    'visualizations',
    'filtertools',
    ]

from .metadata import __author__, __version__
from pyret import *
