'''
stimulustools.py

Tools for basic manipulation of stimulus arrays.

(C) 2014 bnaecker, nirum
'''

import numpy as np

def upsamplestim(stim, upfact, time=None):
    '''

    Upsample the given stimulus by the given factor.

    Input
    -----

    stim (ndarray):
        The actual stimulus to be upsampled.

    upfact (int):
        The upsample factor.

    time (ndarray) [optional]:
        The time axis of the original stimulus.

    Output

    stim_us (ndarray), time_us (ndarray):
        The upsampled time vector and stimulus array

    '''

    # Compute old and new sizes
    oldsz   = stim.shape
    newsz   = oldsz[:-1] + (upfact * oldsz[-1],)

    # Upsample the stimulus array
    stim_us = (stim.reshape((-1, 1)) * np.ones((1, upfact))).reshape(newsz)

    # Upsample the time vecctor if given
    if time is not None:
        x       = np.arange(0, upfact * time.size)
        xp      = np.arange(0, upfact * time.size, upfact)
        time_us = np.interp(x, xp, np.squeeze(time))

    else:
        time_us = None

    return stim_us, time_us

def downsamplestim(stim, downfact, time=None):
    '''

    Downsample the given stimulus by the given factor.

    Input
    -----

    stim (ndarray):
        The original stimulus array

    downfact (int):
        The factor by which the stimulus will be downsampled

    time (ndarray) [optional]:
        The time axis of the original stimulus

    Output
    ------

    stim_ds (ndarray), time_ds (ndarray):
        The downsampled time vector and stimulus array

    '''

    # Downsample the stimulus array
    stim_ds = np.take(stim, np.arange(0, stim.shape[-1], downfact), axis=-1)
    
    # Downsample the time vector, if given
    time_ds = time[::downfact] if time is not None else None

    return stim_ds, time_ds

def slicestim(stim, history, locations=None, tproj=None):
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

    tproj (ndarray) [optional]:
        Matrix of temporal filters to project stimuli onto

    Output
    ------

    slices (ndarray):
        Array of stimulus slices, with all stimulus dimensions collapsed into one. 
        That is, it has shape (np.prod(stim.shape), `history`)

    '''

    # Collapse any spatial dimensions of the stimulus array
    cstim = stim.reshape(-1, stim.shape[-1])

    # Check history is an int
    if history != int(history):
        raise ValueError('"history" must be an integer')
    history = int(history)

    # Compute spatial locations to take
    if locations is None:
        locations = np.ones(cstim.shape[-1])
    
    # Don't include first `history` frames regardless
    locations[:history] = False

    # Construct full stimulus slice array
    if tproj is None:

        # Preallocate
        slices = np.empty((int(history * cstim.shape[0]), int(np.sum(locations[history:]))))

        # Loop over requested time points
        for idx in np.where(locations)[0]:
            slices[:, idx-history] = cstim[:, idx - history :idx].ravel()

    # Construct projected stimulus slice array
    else:

        # Preallocate
        slices = np.empty((int(tproj.shape[1] * cstim.shape[0]), int(np.sum(locations[history:]))))

        # Loop over requested time points
        for idx in np.where(locations)[0]:

            # Project onto temporal basis
            slices[:, idx-history] = (cstim[:, idx-history:idx].dot(tproj)).ravel()

    return slices

def getcov(stim, history, tproj=None, cutoff=0.1):
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
    
    cutoff (default=0.1):
        The cutoff for small singular values in computing the inverse covariance matrix

    tproj (ndarray):
        Temporal basis set to use. Must have # of rows (first dimension) equal to history.
        Each extracted stimulus slice is projected onto this basis set, which reduces the size
        of the corresponding covariance matrix to store.

    Output
    ------

    cov (ndarray):
        (n*n*t by n*n*t) Covariance matrix

    covinv (ndarray):
        (n*n*t by n*n*t) Inverse covariance matrix (computed using the pseudoinverse)

    '''

    # temporal basis (if not given, use the identity matrix)
    if tproj is None:
        tproj = np.eye(history)

    if tproj.shape[0] != history:
        raise ValueError('The first dimension of the basis set tproj must equal history')

    # Collapse any spatial dimensions of the stimulus array
    cstim = stim.reshape(-1, stim.shape[-1])

    # store mean + covariance matrix
    mean = np.zeros(cstim.shape[0] * tproj.shape[1])
    cov = np.zeros((cstim.shape[0] * tproj.shape[1], cstim.shape[0]*tproj.shape[1]))

    # pick some indices to go through
    indices = np.arange(history,cstim.shape[1])
    numpts  = np.min(( cstim.shape[0]*tproj.shape[1]*10, indices.size ))
    np.random.shuffle(indices)

    # loop over temporal indices
    for j in range(numpts):

        # pick which index to use
        idx = indices[j]
        print('[%i of %i]' % (j,numpts))

        # get this stimulus slice, projected onto the basis set tproj
        stimslice = cstim[:, idx - history : idx].dot(tproj).reshape(-1,1)

        # update the mean
        mean += np.squeeze(stimslice)

        # add it to the covariance matrix
        cov += stimslice.dot(stimslice.T)

    # normalize and compute the mean outer product
    mean = mean / (cstim.shape[1] - history)
    mean_op = mean.reshape(-1,1).dot(mean.reshape(1,-1))

    # mean-subtract and normalize the STC by the number of points
    cov = (cov / (cstim.shape[1]-history)) - mean_op

    # compute the inverse covariance
    try:
        covinv = np.linalg.pinv(cov, cutoff)
    except np.linalg.LinAlgError:
        print('Warning: could not compute the inverse covariance.')
        covinv = None

    return cov, covinv
