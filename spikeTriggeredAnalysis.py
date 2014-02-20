import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage.filters as scipyfilt
from scipy.linalg import svd
from scipy.linalg import diagsvd
from helpers.stimulusTools import upsampleStimulus

"""
spike triggered analyses
author: Niru Maheswaranathan
06:55 PM Jan 9, 2014
"""

def computeSTE(time, stimulus, spikes, filterLength):
    """
    collects spike-triggered stimuli:
    given a stimulus and set of spike times, returns a matrix of spike-triggered stimuli

    input
    -----
    time is a (t by 1) time vector corresponding to the stimulus array
    stimulus is a (p by p by t) array where (p by p) is the spatial dimension and t is the number of time points
    spikes is an array of spike times
    filterLength is how many samples to keep in the stimulus slice (with sampling rate defined by time)

    output
    ------
    the STE is returned as a (d by m) matrix, where d is the dimensionality of the
    stimulus slice and m is the number of spikes
    """

    # bin spikes
    (hist,bins) = np.histogram(spikes, time)

    # get indices of non-zero firing
    nzhist = np.where(hist > 0)[0]
    nzhist = nzhist[nzhist > filterLength]

    # create the STE array
    ste = np.empty(stimulus.shape[:2] + (filterLength,nzhist.size))

    # pick out times when spike count is non-zero and add the stimulus slices to the STE
    for idx,val in enumerate(nzhist):
        ste[:,:,:,idx] = hist[val] * stimulus[:,:,val-filterLength:val]

    return ste

def computeSTA(time, stimulus, spikes, filterLength):
    """
    computes a spike-triggered average:
    given a stimulus and set of spike times, returns the spike-triggered average

    input
    -----
    time is a (t by 1) time vector corresponding to the stimulus array
    stimulus is a (p by p by t) array where (p by p) is the spatial dimension and t is the number of time points
    spikes is an array of spike times
    filterLength is how many samples to keep in the stimulus slice (with sampling rate defined by time)

    output
    ------
    the STA is returned as a (d by 1) array, where d is the dimensionality of the stimulus slice
    """

    # bin spikes
    (hist,bins) = np.histogram(spikes, time)

    # get indices of non-zero firing
    nzhist = np.where(hist > 0)[0]
    nzhist = nzhist[nzhist > filterLength]

    # store the STA
    sta = np.empty(stimulus.shape[:2] + (filterLength,))

    # pick out times when spike count is non-zero and add the stimulus slices to the STA
    for idx in nzhist:
        sta += stimulus[:,:,(idx-filterLength):idx] * hist[idx]

    return sta, float(np.sum(hist[nzhist]))

def findFilterPeak(sta):
    """
    finds the peak (single point in space and time) of a smoothed filter
    """

    # smooth the filter a bit first
    fs = smoothFilter(sta, spatialSigma=0.7, temporalSigma=1)

    # find the maximum of |F|
    idx = np.unravel_index(np.abs(fs).argmax(), fs.shape)

    # split into spatial / temporal indices
    spatialIdx = np.roll(np.array(idx[:2]),1)
    temporalIdx = idx[-1]

    return idx, spatialIdx, temporalIdx

def reduceSpatialProfile(stim, idx, width=5):
    """
    Reduces the spatial dimensionality of a stimulus / filter
    """

    # find block of indices
    rowIndices = np.arange(idx[0]-width,idx[0]+width+1)
    colIndices = np.arange(idx[1]-width,idx[1]+width+1)

    # make sure they fit in the full window
    rowIndices = rowIndices[(rowIndices >= 0) & (rowIndices < stim.shape[0])]
    colIndices = colIndices[(colIndices >= 0) & (colIndices < stim.shape[1])]

    # form the mesh
    rmesh, cmesh = np.meshgrid(rowIndices, colIndices)

    # extract the reduced spatial profile
    return stim[rmesh, cmesh, :]

def smoothFilter(f, spatialSigma=0.5, temporalSigma=1):
    """
    fsmooth = smoothSTA(f, spatialSigma=1, temporalSigma=1.5)

    smooths a 3D spatiotemporal filter using a multi-dimensional Gaussian filter

    input
    -----
    f:                  3-D filter to be smoothed
    spatialSigma:       std. dev. of gaussian in the spatial domain
    temporalSigma:      std. dev. of gaussian in the temporal domain

    output
    ------
    fsmooth:            the smoothed filter, has the same shape as f
    """

    return scipyfilt.gaussian_filter(f, (spatialSigma, spatialSigma, temporalSigma), order=0)

def lowranksta(f, k=10):
    """
    fk, u, s, v = lowranksta(f, k=10)

    Decomposes a 3D spatiotemporal filter into the outer product of spatial and temporal components (via the SVD)

    input
    -----
    f:         3-D filter to be separated
    k:         number of components to keep (rank of the filter)

    output
    ------
    fk:        the rank-k filter
    u:         the top k spatial components  (each row is a component)
    s:         the top k singular values
    u:         the top k temporal components (each column is a component)
    """

    # use the SVD to decompose the filter into spatial and temporal components
    u,s,v = svd(f.reshape(-1,f.shape[-1]), full_matrices=False)

    # choose how many components to keep
    k = np.min([k,s.size])

    # create the rank-k filter
    fk = (u[:,:k].dot(np.diag(s[:k]).dot(v[:k,:]))).reshape(f.shape)

    # return these components and the low-rank STA
    return fk, u, s, v

def decompose(sta):
	'''
	Usage: s, t = decompose(sta)
	Decomposes a spatiotemporal STA into a spatial and temporal kernel

	'''
	_, u, _, v = lowranksta(f, k=1)
	return u[:, 0].reshape(sta.shape[:2]), v[:, 0]

