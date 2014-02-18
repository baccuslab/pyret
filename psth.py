import numpy as np
import scipy.signal as sps

"""
PSTH analysis tools
author: Niru Maheswaranathan
07:00 PM Jan 7, 2014
"""

def smoothPSTH(bins, psth, lpfFreq=10):
    # smooth by low-pass filtering
    fs = 1/np.mean(np.diff(bins))   # sampling rate in Hz
    nyquistCutoff = lpfFreq/(0.5*fs)      # represent cutoff frequency as a fraction of the nyquist frequency (0.5*fs)

    # check to make sure the cutoff is between 0 and 1
    if (nyquistCutoff < 0) or (nyquistCutoff > 1):
        raise ValueError('Cutoff frequency specified (%3.2f Hz) falls outside the valid range defined by the Nyquist rate (0-%3.2f Hz).' % (lpfFreq, 0.5*fs))

    # create the 1st order butterworth filter
    b, a = sps.butter(1, nyquistCutoff)

    # smooth the PSTH
    smoothPSTH = sps.filtfilt(b,a,psth)

    return smoothPSTH

def computePSTH(spk, binWidth=10e-3, **kwargs):
    """
    Computes a peri-stimulus time histogram (PSTH)

    Usage
    -----
    (time,psth) = psth.computePSTH(spikes)

    Parameters
    -----
    spikes    : an (m x 2) numpy array which contains spike times in the first column and trial indices in the second
    binWidth  : size of bins, in milliseconds (default is 10e-3 seconds or 10ms)
    tmax      : the PSTH stops when this time is reached (seconds, default is the time of the last spike)
    tmax      : the PSTH stops when this time is reached (seconds, default is the time of the last spike)
    density   : whether or not to normalize the PSTH to be a probability density function (PDF) (default is False)

    Returns
    -----
    time: an array of bins used to bin spike times
    psth: the values of the PSTH at each time bin

    """

    # reshape spikes if necessary
    if np.ndim(spk) > 1:
        spk = np.squeeze(spk)

    # if a time vector given
    if 'time' in kwargs.keys():
        psth, bins = np.histogram(spk, bins=kwargs['time'])

    else:

        # if no maximum time is given, set it equal to the time of the last spike
        if 'tmax' not in kwargs.keys():
            tmax = np.ceil(np.max(spk))
        else:
            tmax = kwargs['tmax']

        # compute PSTH via numpy's histogram
        psth, bins = np.histogram(spk, bins=np.arange(0,tmax+binWidth,binWidth))

    # normalize if necessary
    #if 'density' in kwargs.keys():
        #psth = psth/float(np.unique(spk[:,1]).size)

    # get bin centers (times)
    centeredBins = np.diff(bins)/2 + bins[:-1]

    return centeredBins, psth
