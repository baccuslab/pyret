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
containers      - Objects for managing experimental data (stimuli and spikes)

For more information, see the accompanying README.md

"""

__all__ = [
    'spiketools',
    'nonlinearities',
    'stimulustools',
    'visualizations',
    'filtertools',
    'containers'
    ]

__version__ = '0.4.0'

from pyret import *
