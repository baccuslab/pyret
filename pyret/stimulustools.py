"""
Tools for dealing with spatiotemporal stimuli

"""

import numpy as np
from scipy.linalg.blas import get_blas_funcs

__all__ = ['upsample_stim', 'downsample_stim', 'slicestim', 'getcov']


def upsample_stim(stim, upsample_factor, time=None):
    """
    Upsample the given stimulus by the given factor.

    Parameters
    ----------
    stim : array_like
        The actual stimulus to be upsampled. dimensions: (time, space, space)

    upsample_factor : int
        The upsample factor.

    time : arra_like, optional
        The time axis of the original stimulus.

    Returns
    -------
    stim_us : array_like
        The upsampled stimulus array

    time_us : array_like
        the upsampled time vector

    """

    # Upsample the stimulus array
    stim_us = np.repeat(stim, upsample_factor, axis=0)

    # if time vector is not given
    if time is None:
        return stim_us, None

    # Upsample the time vecctor if given
    x = np.arange(0, upsample_factor * time.size)
    xp = np.arange(0, upsample_factor * time.size, upsample_factor)
    time_us = np.interp(x, xp, np.squeeze(time))

    # Check that last k timestamps are valid. np.interp does no
    # extrapolation, which may be necessary for the last
    # timepoint, given the method above
    modified_time_us = time_us.copy()
    dt = np.diff(time_us).mean()
    for k in reversed(np.arange(upsample_factor) + 1):
        if np.allclose(time_us[-(k+1)], time_us[-k]):
            modified_time_us[-k] = modified_time_us[-(k+1)] + dt
    time_us = modified_time_us.copy()

    return stim_us, time_us


def downsample_stim(stim, downsample_factor, time=None):
    """
    Downsample the given stimulus by the given factor.

    Parameters
    ----------
    stim : array_like
        The original stimulus array

    downsample_factor : int
        The factor by which the stimulus will be downsampled

    time : array_like, optional
        The time axis of the original stimulus

    Returns
    -------
    stim_ds : array_like
        The downsampled stimulus array

    time_ds : array_like
        The downsampled time vector

    """

    # Downsample the stimulus array
    stim_ds = np.take(stim, np.arange(0, stim.shape[-1], downsample_factor), axis=-1)

    # Downsample the time vector, if given
    time_ds = time[::downsample_factor] if time is not None else None

    return stim_ds, time_ds


def slicestim(stimulus, history, locations=None, tproj=None):
    """
    Slices a spatiotemporal stimulus array (over time) into overlapping frames.

    Parameters
    ----------
    stimulus : array_like
        The spatiotemporal or temporal stimulus to slices. Should have shape
        (n, n, t) or (t,).

    history : int
        Integer number of time points to keep in each slice.

    locations : array_like of booleans
        Boolean array of temporal locations at which slices are taken. If unspecified,
        use all time points.

    tproj : array_like, optional
        Matrix of temporal filters to project stimuli onto

    Returns
    ------
    slices : array_like
        Array of stimulus slices, with all stimulus dimensions collapsed into one.
        That is, it has shape (np.prod(stimulus.shape), `history`)

    """

    # Collapse any spatial dimensions of the stimulus array
    cstim = stimulus.reshape(-1, stimulus.shape[-1])

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
            slices[:, idx - history] = cstim[:, idx - history:idx].ravel()

    # Construct projected stimulus slice array
    else:

        # Preallocate
        slices = np.empty((int(tproj.shape[1] * cstim.shape[0]), int(np.sum(locations[history:]))))

        # Loop over requested time points
        for idx in np.where(locations)[0]:

            # Project onto temporal basis
            slices[:, idx - history] = (cstim[:, idx - history:idx].dot(tproj)).ravel()

    return slices


def getcov(stimulus, history, tproj=None, verbose=False):
    """
    Computes a stimulus covariance matrix

    .. warning:: This is computationally expensive for large stimuli

    Parameters
    ----------
    stimulus : array_like
        The spatiotemporal or temporal stimulus to slices. Should have shape
        (n, n, t) or (t,).

    history : int
        Integer number of time points to keep in each slice.

    tproj : array_like, optional
        Temporal basis set to use. Must have # of rows (first dimension) equal to history.
        Each extracted stimulus slice is projected onto this basis set, which reduces the size
        of the corresponding covariance matrix to store.

    verbose : boolean, optional
        If True, print out progress of the computation. (defaults to False)

    Returns
    ------
    stim_cov : array_like
        (n*n*t by n*n*t) Covariance matrix

    """

    # temporal basis (if not given, use the identity matrix)
    if tproj is None:
        tproj = np.eye(history)

    if tproj.shape[0] != history:
        raise ValueError('The first dimension of the basis set tproj must equal history')

    # Collapse any spatial dimensions of the stimulus array
    cstim = stimulus.reshape(-1, stimulus.shape[-1])

    # store mean + covariance matrix
    mean = np.zeros(cstim.shape[0] * tproj.shape[1])
    stim_cov = np.zeros((cstim.shape[0] * tproj.shape[1], cstim.shape[0] * tproj.shape[1]))

    # pick some indices to go through
    indices = np.arange(history,cstim.shape[1])
    numpts  = np.min(( cstim.shape[0] * tproj.shape[1] * 10, indices.size ))
    np.random.shuffle(indices)

    # get blas function
    blas_ger_fnc = get_blas_funcs(('ger',), (stim_cov,))[0]

    # loop over temporal indices
    for j in range(numpts):

        # pick which index to use
        idx = indices[j]
        if verbose:
            if np.mod(j,100) == 0:
                print('[%i of %i]' % (j,numpts))

        # get this stimulus slice, projected onto the basis set tproj
        stimslice = cstim[:, idx - history:idx].dot(tproj).reshape(-1,1)

        # update the mean
        mean += np.squeeze(stimslice)

        # add it to the covariance matrix (using low-level BLAS operation)
        blas_ger_fnc(1, stimslice, stimslice, a=stim_cov.T, overwrite_a=True)

    # normalize and compute the mean outer product
    mean = mean / numpts
    mean_op = mean.reshape(-1,1).dot(mean.reshape(1,-1))

    # mean-subtract and normalize the STC by the number of points
    stim_cov = (stim_cov / (numpts - 1)) - mean_op

    return stim_cov
