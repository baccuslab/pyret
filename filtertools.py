'''
filtertools.py

tools for computation of basic linear filters

(C) 2014 bnaecker, nirum
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.ndimage.filters import gaussian_filter

def getste(time, stimulus, spikes, filterlength):
    '''
    Usage: ste, tax = getste(time, stimulus, spikes, filterlength)
    Construct the spike-triggered ensemble

    Input
    -----

    time:
        The time axis of the stimulus

    stimulus:
        The stimulus array. The first dimension of the stimulus
        should be the time axis (and be of the same length as
        the `time` array), but no other restrictions are placed
        on the shape of the array. In other words, it works for
        temporal or spatiotemporal stimuli.

    spikes:
        Array of spike times.

    filterlength:
        Integer number of frames over which to construct the
        ensemble

    Output
    ------

    ste:
        The spike-triggered stimulus ensemble. The returned array
        has stimulus.ndim + 1 dimensions, and has a shape of
        (nspikes, filterlength, stimulus.shape[0]).

    tax:
        The time axis of the ensemble. It is of length `filterlength`,
        with intervals given by the sample rate of the `time` input
        array.

    '''

    # Bin spikes
    (hist, bins) = np.histogram(spikes, time)

    # Get indices of non-zero firing, truncating spikes earlier
    # than `filterlength` frames
    nzhist = np.where(hist > 0)[0]
    nzhist = nzhist[nzhist > filterlength]

    # Collapse any spatial dimensions of the stimulus array
    cstim = stimulus.reshape(stimulus.shape[0], -1)

    # Preallocate STE array
    ste = np.empty((nzhist.size, filterlength, cstim.shape[0]))

    # Add filterlength frames preceding each spike to the STE array
    for idx, val in enumerate(nzhist):
        ste[idx, :, :] = hist[val] * stimulus[val - filterlenth : val, :]

    # Construct a time axis to return
    tax = time[:filterlength] - time[0]

    # Return reshaped STE and the time axis
    return ste.reshape(nzhist.size, stimulus.shape), tax

def getsta(time, stimulus, spikes, filterlength):
    '''
    Usage: sta, tax = getste(time, stimulus, spikes, filterlength)
    Compute the spike-triggered average

    Input
    -----

    time:
        The time axis of the stimulus

    stimulus:
        The stimulus array. The first dimension of the stimulus
        should be the time axis (and be of the same length as
        the `time` array), but no other restrictions are placed
        on the shape of the array. In other words, it works for
        temporal or spatiotemporal stimuli.

    spikes:
        Array of spike times.

    filterlength:
        Integer number of frames over which to construct the
        ensemble

    Output
    ------

    sta:
        The spike-triggered average. The returned array has 
        stimulus.ndim + 1 dimensions, and has a shape of
        (nspikes, stimulus.shape[0], filterlength).

    tax:
        The time axis of the ensemble. It is of length `filterlength`,
        with intervals given by the sample rate of the `time` input
        array.

    '''

    # Bin spikes
    (hist, bins) = np.histogram(spikes, time)

    # Get indices of non-zero firing, truncating spikes earlier
    # than `filterlength` frames
    nzhist = np.where(hist > 0)[0]
    nzhist = nzhist[nzhist > filterlength]

    # Collapse any spatial dimensions of the stimulus array
    cstim = stimulus.reshape(stimulus.shape[0], -1)

    # Preallocate STA array
    sta = np.empty((filterlength, cstim.shape[1]))

    # Add filterlength frames preceding each spike to the running STA
    for idx in nzhist:
        sta += hist[idx] * stimulus[idx - filterlength : idx, :]

    # Construct a time axis to return
    tax = time[:filterlength] - time[0]

    # Return reshaped STA and the time axis
    return sta.reshape(filterlength, stimulus.shape[1:]), tax

def lowranksta(f, k=10):
    '''
    Usage: fk, u, s, v = lowranksta(f, k=10)
    Decomposes a 3D spatiotemporal filter into the outer product 
    of spatial and temporal components (via the SVD). This is useful,
    for example, in computing the spatial and temporal kernels of a
    spatiotemporal STA, or in denoising the STA.

    Input
    -----

    f:
        3-D filter to be separated

    k:
        number of components to keep (rank of the filter)

    Output
    ------

    fk:
        the rank-k filter

    u:
        the top k spatial components  (each row is a component)

    s:
        the top k singular values

    u:
        the top k temporal components (each column is a component)

    '''

    # Compute the SVD of the full filter
    u, s, v = svd(f.reshape(-1, f.shape[-1]), full_matrices=False)

    # Keep the top k components
    k = np.min([k, s.size])

    # Compute the rank-k filter
    fk = (u[:,:k].dot(np.diag(s[:k]).dot(v[:k,:]))).reshape(f.shape)

    # Return the rank-k approximate filter, and the SVD components
    return fk, u, s, v

def decompose(sta):
    '''
    Usage: s, t = decompose(sta)
    Decomposes a spatiotemporal STA into a spatial and temporal kernel

    Input
    -----

    sta:
        The full 3-dimensional STA to be decomposed

    Output
    ------

    s:
        The spatial kernel

    t:
        The temporal kernel

    '''
    _, u, _, v = lowranksta(f, k=1)
    return u[:, 0].reshape(sta.shape[:2]), v[:, 0]

def _fit2Dgaussian(histogram, numSamples=1e4):
    ''' Fit 2D gaussian to empirical histogram '''

    # Indices
    x = np.linspace(0,1,histogram.shape[0])
    y = np.linspace(0,1,histogram.shape[1])
    xx,yy = np.meshgrid(x,y)

    # Draw samples
    indices = np.random.choice(np.flatnonzero(histogram+1), size=numSamples, replace=True, p=histogram.ravel())
    x_samples = xx.ravel()[indices]
    y_samples = yy.ravel()[indices]

    # Fit mean / covariance
    samples = np.array((x_samples,y_samples))
    centerIdx = np.unravel_index(np.argmax(histogram), histogram.shape)
    center = (xx[centerIdx], yy[centerIdx])
    C = np.cov(samples)

    # Get width / angles
    widths,vectors = np.linalg.eig(C)
    angle = np.arccos(vectors[0,0])

    return center, widths, angle, xx, yy

def _im2hist(data, spatialSmoothing = 2.5):
    ''' Converts 2D image to histogram '''

    # Smooth the data
    data_smooth = gaussian_filter(data, spatialSmoothing, order=0)

    # Mean subtract
    mu = np.median(data_smooth)
    data_centered = data_smooth - mu

    # Figure out if it is an on or off profile
    if np.abs(np.max(data_centered)) < np.abs(np.min(data_centered)):

        # flip from 'off' to 'on'
        data_centered *= -1;

    # Min-subtract
    data_centered -= np.min(data_centered)

    # Normalize to a PDF
    pdf = data_centered / np.sum(data_centered)

    return pdf

def getellipse(F, scale=1.5):
    '''
    Usage: ell = getellipse(staframe, scale=1.5)
    Fit an ellipse to the given spatial receptive field

    Input
    -----

    staframe:
            The spatial receptive field to which the ellipse should be fit

    scale:
            Scale factor for the ellipse

    Output
    ------

    ell:
            A matplotlib.patches.Ellipse object

    '''

    # Get ellipse parameters
    histogram = _im2hist(F)
    center, widths, theta, xx, yy = fit2Dgaussian(histogram, numSamples=1e5)

    # Generate ellipse
    ell = Ellipse(xy=center, width=scale*widths[0], height=scale*widths[1], angle=np.rad2deg(theta)+90)

    return ell

def filterpeak(sta):
    '''
    Usage: idx, spaceidx, timeidx = filterpeak(sta)
    Find the peak (single point in space/time) of a smoothed filter

    '''
    # Smooth filter
    fs = smoothfilter(sta, spacesig=0.7, timesig=1)

    # Find the index of the maximal point
    idx = np.unravel_index(np.abs(fs).argmax(), fs.shape)

    # Split into spatial/temporal indices
    sidx = np.roll(idx[:2], 1)
    tidx = idx[-1]

    # Return the indices
    return idx, sidx, tidx

def smoothfilter(f, spacesig=0.5, timesig=1):
    '''
    Usage: fsmooth = smoothfilter(f, spacesig=0.5, timesig=1):
    Smooths a 3D spatiotemporal linear filter using a multi-dimensional
    Gaussian filter with the given properties.

    Input
    -----

    f:
            3D filter to be smoothed
    
    spacesig, timesig:
            The spatial and temporal standard deviations of the Gaussian
            filter used to smooth the given filter

    Output
    ------

    fsmooth:
            The smoothed filter, with the same shape as the input
    
    '''
    return gaussian_filter(f, (spacesig, spacesig, timesig), order=0)

def cutout(s, idx, width=5):
    '''
    Usage: cut = cutout(s, idx, width=5)
    Cut out a chunk of the given stimulus or filter

    Input
    -----

    s:
        Stimulus or filter from which the chunk is cut out

    idx:
        2D array-like, specifying the row and column indices of
        the center of the section to be cut out

    width:
        The size of the chunk to cut out from the start indices

    Output
    ------

    rs:
        The cut out section of the given stimulus or filter

    '''

    # Find the indices
    row = np.arange(idx[0] - width, idx[0] + width + 1)
    col = np.arange(idx[1] - width, idx[1] + width + 1)

    # Make sure the indices are within the bounds of the given array
    row = row[(row >= 0) & (row < stim.shape[0])]
    col = col[(col >= 0) & (col < stim.shape[1])]

    # Mesh the indices
    rmesh, cmesh = np.meshgrid(row, col)

    # Extract and return the reduced array
    return s[rmesh, cmesh, :]

