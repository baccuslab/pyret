'''
spktools.py

Tools for loading and basic manipulation of spike times

(C) 2014 bnaecker, nirum
'''

import os
import re
import scipy as sp
from scipy.io import loadmat
from scipy import signal

def binspikes(spk, tmax, binsize=0.01):
    '''
    Usage: bspk, tax = binspikes(spk, tmax, binsize)
    Bin spike times at the given resolution

    Input
    -----

    spk:
        Array of spike times

    tmax:
        Maximum bin time. Usually end of experiment, but could
        really be anything.

    binsize:
        Size of bins (in milliseconds).

    Output
    ------
    bspk:
        Binned spike times

    tax:
        The bins themselves.

    '''
    # Histogram count for each cell
    tbins = sp.arange(0, tmax, binsize)
    bspk, tax = sp.histogram(cell, bins = tbins)
    return bspk, tax[:-1]

def estfr(bspk, binsize=0.01, npts=7, sd=2):
    '''
    Usage: rates = estfr(bspk, npts, sd)
    Estimate the instantaneous firing rates from binned spike counts

    Input
    -----
    bspk:
        Array of binned spike counts (as from binspikes)

    npts:
        Number of points in Gaussian filter used to smooth counts

    sd:
        SD (in points) of the Gaussian filter used to smooth counts

    Output
    ------

    rates:
        Array of estimated instantaneous firing rate

    '''
    # Construct Gaussian filter
    filt = signal.gaussian(npts, sd)

    # Filter  binned spike times
    return signal.lfilter(filt, 1, cell) / binsize
