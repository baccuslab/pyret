'''
spiketools.py

Tools for basic manipulation of spike trains

(C) 2014 bnaecker, nirum
'''

import numpy as np
from scipy.io import loadmat
from scipy import signal

def binspikes(spk, tmax=None, binsize=0.01, time=None):
    '''
    
    Bin spike times at the given resolution. The function has two forms.

    Input
    -----

    spk (ndarray):
        Array of spike times
	
    EITHER:

        tmax (float):
            Maximum bin time. Usually end of experiment, but could
            really be anything.

        binsize (float):
            Size of bins (in milliseconds).

    OR:

        time (ndarray):
            The array to use as the actual bins to np.histogram

    Output
    ------

    bspk (ndarray):
        Binned spike times

    tax (ndarray):
        The bin centers

    '''

    # Check if actual time bins are specified
    if time is not None:
        return np.histogram(spk, bins=time)

    # If not, use either tmax or the maximum spike time and the binsize
    if not tmax:
        tmax = spk.max()
    tbins   = np.arange(0, tmax, binsize)
    bspk, _ = np.histogram(spk, bins=tbins)

    return bspk, tbins + binsize / 2

def estfr(bspk, binsize=0.01, npts=7, sd=2):
    '''
    
    Estimate the instantaneous firing rates from binned spike counts

    Input
    -----

    bspk (ndarray):
        Array of binned spike counts (as from binspikes)

    npts (int):
        Number of points in Gaussian filter used to smooth counts

    sd (int):
        SD (in points) of the Gaussian filter used to smooth counts

    Output
    ------

    rates (ndarray):
        Array of estimated instantaneous firing rate

    '''
    # Construct Gaussian filter
    filt = signal.gaussian(npts, sd)

    # Filter  binned spiketimes
    return signal.filtfilt(filt, 1, bspk) / binsize
