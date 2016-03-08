"""
Tools for dealing with spatiotemporal stimuli

"""

import warnings

import numpy as np
from scipy.linalg.blas import get_blas_funcs

__all__ = ['upsample_stim', 'downsample_stim', 'slicestim', 'getcov',
           'rolling_window']


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
        if np.allclose(time_us[-(k + 1)], time_us[-k]):
            modified_time_us[-k] = modified_time_us[-(k + 1)] + dt
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
    stim_ds = np.take(stim, np.arange(0, stim.shape[0], downsample_factor), axis=0)

    # Downsample the time vector, if given
    time_ds = time[::downsample_factor] if time is not None else None

    return stim_ds, time_ds


def slicestim(stimulus, history):
    """
    Slices a spatiotemporal stimulus array (over time) into overlapping frames.

    Parameters
    ----------
    stimulus : array_like
        The spatiotemporal or temporal stimulus to slices. Should have shape
        (t, ...), so that the time axis is first. The ellipses indicate the 
        spatial dimensions of the stimulus, if any.

    history : int
        Integer number of time points to keep in each slice.

    Returns
    ------
    slices : array_like
        A view onto the original stimulus array, giving the overlapping slices
        of the stimulus. The full shape of the returned array is:
        (history, stimulus.shape[0] - history, ...). As above, the ellipses
        indicate any spatial dimensions to the stimulus.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """

    if not (1 <= history <= stimulus.shape[0]):
        raise ValueError(
                '`history` must be between 1 and {0:#d}'.format(
                stimulus.shape[0]))
    history = int(history)

    # Use strides to create view onto array
    shape = (history, stimulus.shape[0] - history) + stimulus.shape[1:]
    strides = (stimulus.strides[0],) + stimulus.strides
    return np.lib.stride_tricks.as_strided(stimulus, 
            shape=shape, strides=strides)


def getcov(stimulus, history, tproj=None, verbose=False):
    """
    Computes a stimulus covariance matrix

    .. warning:: This is computationally expensive for large stimuli

    Parameters
    ----------
    stimulus : array_like
        The spatiotemporal or temporal stimulus to slices. Should have shape
        (t, ...), where the ellipses indicate any spatial dimensions.

    history : int
        Integer number of time points to keep in each slice.

    tproj : array_like, optional
        Temporal basis set to use. Size of the second dimension must be equal
        to `history`. Each extracted stimulus slice is projected onto this 
        basis set, which reduces the size of the corresponding covariance 
        matrix to store.

    verbose : boolean, optional
        If True, print out progress of the computation. (defaults to False)

    Returns
    ------
    stim_cov : array_like
        (n*n*t by n*n*t) Covariance matrix

    """
    if not (1 <= history < stimulus.shape[0]):
        raise ValueError('`history` must be in [1, {0:#d})'.format(
                stimulus.shape[0]))

    # Collapse spatial dimensions
    cstim = stimulus.reshape(stimulus.shape[0], -1)

    # store mean + covariance matrix
    mean = np.zeros(cstim.shape[-1] * history)
    stim_cov = np.zeros((cstim.shape[-1] * history,) * 2)

    # Update covariance matrix from random points in the stimulus
    indices = np.arange(history, cstim.shape[0])
    numpts = np.min((cstim.shape[1] * 10, indices.size))
    np.random.shuffle(indices)

    # BLAS rank-1 covariance update function
    blas_ger_fnc = get_blas_funcs(('ger',), (stim_cov,))[0]

    for j in range(numpts):
        idx = indices[j]
        if verbose and np.mod(j, 100) == 0:
            print('[%i of %i]' % (j, numpts))

        stimslice = np.reshape(cstim[idx - history : idx, :], 
                (-1, 1), order='F')

        mean += np.squeeze(stimslice)

        # Update covariance matrix
        blas_ger_fnc(1, stimslice, stimslice, 
                a=stim_cov.T, overwrite_a=True)

    # Normalize and compute the mean outer product
    mean /= numpts
    mean_op = mean.reshape(-1, 1).dot(mean.reshape(1, -1))

    # Return normalized covariance matrix
    stim_cov = (stim_cov / (numpts - 1)) - mean_op

    return stim_cov


def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])

    """
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.warn('`rolling_window` is deprecated and will be removed' +
                ' in future releases. Use `stimulustools.slicestim` instead.',
                DeprecationWarning)
    return slicestim(array, window)
