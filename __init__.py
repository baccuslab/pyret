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

__all__ = ['spiketools',
           'filtertools',
           'stimulustools',
           'visualizations',
           'nonlinearities']
__version__ = '0.2'

from pyret import *
