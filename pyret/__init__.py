"""
Pyret
=====

A Python package for analysis of retinal data

Modules
-------
spiketools      - Tools for manipulating spike trains
filtertools     - Analysis of linear filters
stimulustools   - Tools for getting stimulus history and statistics
visualizations  - Tools for plotting receptive fields, filters, etc.
nonlinearities  - Methods for estimating nonlinearities

For more information, see the accompanying README.md

"""

__all__ = [
    'spiketools',
    'nonlinearities',
    'stimulustools',
    'visualizations',
    'filtertools',
    ]

__version__ = '0.5.1'

from pyret import *
