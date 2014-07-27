"""
Pyret: A Python package for analysis of retinal data.
(C) 2014 Baccus lab
===================

Modules
-------
    binary          - Interact with *.bin raw experiment data files
    spiketools      - Tools for manipulating spike trains
    filtertools     - Analysis of linear filters
    stimulustools   - Tools for getting stimulus history and statistics
    visualizations  - Tools for plotting receptive fields, filters, etc.
    nonlinearities  - Methods for estimating nonlinearities

Classes
-------
    cell            - An object to hold a cell's data (in development)

For more information, see the accompanying README.md
"""

__all__ = [
        'binary', 
        'spiketools', 
        'filtertools', 
        'stimulustools', 
        'visualizations', 
        'cell', 
        'nonlinearities']
__version__ = '0.1.1'

from pyret import *
