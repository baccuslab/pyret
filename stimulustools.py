'''
stimulustools.py

Tools for basic manipulation of stimulus arrays.

(C) 2014 bnaecker, nirum
'''

import numpy as np

def upsamplestim(time, stim, upfact):
    '''

    Upsample the given stimulus by the given factor.

    Input
    -----

    time (ndarray):
        The time axis of the original stimulus.

    stim (ndarray):
        The actual stimulus to be upsampled.

    upfact (int):
        The upsample factor.

    Output

    time_us (ndarray), stim_us (ndarray):
        The upsampled time vector and stimulus array

    '''

    # Compute old and new sizes
    oldsz   = stim.shape
    newsz   = oldsz[:-1] + (upfact * oldsz[-1],)

    # Upsample the stimulus array
    stim_us = (stim.reshape((-1, 1)) * np.ones((1, upfact))).reshape(newsz)

    # Upsample the time vecctor
    x       = np.arange(0, upfact * time.size)
    xp      = np.arange(0, upfact * time.size, 2)
    time_us = np.interp(x, xp, time)

    return time_us, stim_us

def downsamplestim(time, stim, downfact):
    '''

    Downsample the given stimulus by the given factor.

    Input
    -----

    time (ndarray):
        The time axis of the original stimulus

    stim (ndarray):
        The original stimulus array

    downfact (int):
        The factor by which the stimulus will be downsampled

    Output
    ------

    time_ds (ndarray), stim_ds (ndarray):
        The downsampled time vector and stimulus array

    '''

    # Downsample the stimulus array
    stim_ds = np.take(stim, np.arange(0, stim.shape[-1], downfact), axis=-1)
    
    # Downsample the time vector
    time_ds = time[::downfact]

    return time_ds, stim_ds

def slicestim(stim, history, locations=None):
    '''

    Slices a spatiotemporal stimulus array (over time) into overlapping frames.

    Input
    -----

    stim (ndarray):
        The spatiotemporal or temporal stimulus to slices. Should have shape
        (n, n, t) or (t,).

    history (int):
        Integer number of time points to keep in each slice.

    locations (boolean) [optional]:
        Boolean array of temporal locations at which slices are taken. If unspecified,
        use all time points.

    Output
    ------

    slices (ndarray):
        Array of stimulus slices, with all stimulus dimensions collapsed into one. 
        That is, it has shape (np.prod(stim.shape), `history`)

    '''

    # Converty purely temporal stimuli into spatiotemporal
    if stim.ndim == 1:
        stim = stim.reshape(1, 1, -1)

    # Compute spatial locations to take
    if locations is None:
        locations = np.ones(stim.shape[-1])

    # Preallocate array to hold all slices
    slices  = np.empty((history * np.prod(stim.shape[:2]), np.sum(locations[history:])))
    sidx    = 0

    # Loop over locations (can't use np.take, since we need to keep `history`)
    for idx in range(history, locations.size):
        if locations[idx]:
            slices[:, sidx] = stim[:, :, idx - history :idx].ravel()

    return slices

def getcov(stim, history):
    '''

    Computes a stimulus covariance matrix
    ** Warning: this is computationally expensive for large stimuli **

    Input
    -----

    stim (ndarray):
        The spatiotemporal or temporal stimulus to slices. Should have shape
        (n, n, t) or (t,).

    history (int):
        Integer number of time points to keep in each slice.

    Output
    ------

    cov (ndarray):
        (n*n*t by n*n*t) Covariance matrix

    covinv (ndarray):
        (n*n*t by n*n*t) Inverse covariance matrix (computed using the pseudoinverse)

    '''

    cov    = np.cov(slicestim(stim, history))
    covinv = np.linalg.inv(cov)

    return cov, covinv
