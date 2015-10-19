"""
Tools ansd utilities for computing spike-triggered averages (filters),
finding spatial and temporal components of
spatiotemporal filters, and basic filter signal processing.

"""

import numpy as np
from matplotlib.patches import Ellipse
from numpy.linalg import LinAlgError
from scipy.linalg.blas import get_blas_funcs
from scipy import ndimage
from scipy.stats import skew
from skimage.restoration import denoise_tv_bregman
from skimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from functools import reduce
from warnings import warn

__all__ = ['getste', 'getsta', 'getstc', 'lowranksta', 'decompose',
           'get_ellipse_params', 'fit_ellipse', 'filterpeak', 'smoothfilter',
           'cutout', 'prinangles', 'rolling_window']


def dimension_warning(stim):
    """
    Warning for mis-shaped stimuli (due to the time axis flip in pyret v0.3.1)
    """
    if np.argmax(stim.shape) != 0:
        warn('''Your stimulus seems to have the wrong shape.
             Check to make sure that the time dimension is the first dimension
             (new in v0.3.1)''', DeprecationWarning, stacklevel=2)


def getste(time, stimulus, spikes, filter_length):
    """
    Constructs an iterator over spike-triggered stimuli

    Parameters
    ----------
    time : ndarray
        The time array corresponding to the stimulus

    stimulus : ndarray
        A spatiotemporal or temporal stimulus array
        (where time is the first dimension)

    spikes : iterable
        A list or ndarray of spike times

    filter_length : int
        The desired temporal history / length of the STA

    Returns
    -------
    ste : generator
        A generator that yields samples from the spike-triggered ensemble

    """

    dimension_warning(stimulus)

    # Bin spikes
    (hist, bins) = np.histogram(spikes, time)

    # Get indices of non-zero firing, truncating spikes earlier
    # than `filterlength` frames
    slices = (stimulus[(idx - filter_length):idx, ...].astype('float64')
              for idx in np.where(hist > 0)[0] if idx > filter_length)

    # return the iterator
    return slices


def getsta(time, stimulus, spikes, filter_length):
    """
    Compute a spike-triggered average

    sta, tax = getsta(time, stimulus, spikes, filter_length)

    Parameters
    ----------
    time : ndarray
        The time array corresponding to the stimulus

    stimulus : ndarray
        A spatiotemporal or temporal stimulus array
        (where time is the first dimension)

    spikes : iterable
        A list or ndarray of spike times

    filter_length : int
        The desired temporal history / length of the STA

    Returns
    -------
    sta : ndarray
        The spatiotemporal spike-triggered average (RF)

    tax : ndarray
        A time axis corresponding to the STA

    """

    dimension_warning(stimulus)

    # get the iterator
    ste = getste(time, stimulus, spikes, filter_length)

    # reduce
    sta = reduce(lambda sta, x: np.add(sta, x), ste) / float(len(spikes))

    # time axis
    tax = time[:filter_length] - time[0]

    return sta, tax


def getstc(time, stimulus, spikes, filter_length):
    """
    Compute the spike-triggered covariance

    stc = getstc(time, stimulus, spikes, filter_length)

    Parameters
    ----------
    time : ndarray
        The time array corresponding to the stimulus
        (where time is the first dimension)

    stimulus : ndarray
        A spatiotemporal or temporal stimulus array

    spikes : iterable
        A list or ndarray of spike times

    filter_length : int
        The desired temporal history / length of the STA

    Returns
    -------
    stc : ndarray
        The spike-triggered covariance (STC) matrix

    """

    dimension_warning(stimulus)

    # initialize
    ndims = np.prod(stimulus.shape[1:]) * filter_length
    stc_init = np.zeros((ndims, ndims))

    # get the blas function for computing the outer product
    assert stimulus.dtype == 'float64', 'Stimulus must be double precision'
    outer = get_blas_funcs('syr', dtype='d')

    # get the iterator
    ste = getste(time, stimulus, spikes, filter_length)

    # reduce, note that this only contains the upper triangular portion
    stc_ut = reduce(lambda C, x: outer(1, x.ravel(), a=C),
                    ste, stc_init) / float(len(spikes))

    # make the full STC matrix (copy the upper triangular portion to the lower
    # triangle)
    stc = np.triu(stc_ut, 1).T + stc_ut

    return stc


def lowranksta(f_orig, k=10):
    """
    Constructs a rank-k approximation to the given spatiotemporal filter.
    This is useful for computing a spatial and temporal kernel of an STA,
    or for denoising.

    Parameters
    ----------
    f : array_like
        3-D filter to be separated (time, space, space)

    k : int
        number of components to keep (rank of the filter)

    Returns
    -------
    fk : array_like
        the rank-k filter

    u : array_like
        the top k spatial components  (each row is a component)

    s : array_like
        the top k singular values

    u : array_like
        the top k temporal components (each column is a component)

    """

    # work with a copy of the filter (prevents corrupting the input)
    f = f_orig.copy()

    # Compute the SVD of the full filter
    try:
        assert f.ndim >= 2, "Filter must be at least 2-D"
        u, s, v = np.linalg.svd(f.reshape(f.shape[0], -1) - np.mean(f),
                                full_matrices=False)
    except LinAlgError:
        err = '''The SVD did not converge for the given spatiotemporal filter
              The data is likely too noisy to compute a rank-{0} approximation,
              try reducing the requested rank.'''.format(k)
        raise LinAlgError(err)

    # Keep the top k components
    k = np.min([k, s.size])

    # Compute the rank-k filter
    fk = (u[:, :k].dot(np.diag(s[:k]).dot(v[:k, :]))).reshape(f.shape)

    # make sure the temporal kernels have the correct sign

    # get out the temporal filter at the RF center
    peakidx = filterpeak(f)[1]
    tsta = f[:, peakidx[1], peakidx[0]].reshape(-1, 1)
    tsta -= np.mean(tsta)

    # project onto the temporal filters and keep the sign
    signs = np.sign((u - np.mean(u, axis=0)).T.dot(tsta))

    # flip signs according to this projection
    v *= signs
    u *= signs.T

    # Return the rank-k approximate filter, and the SVD components
    return fk, u, s, v


def decompose(sta):
    """
    Decomposes a spatiotemporal STA into a spatial and temporal kernel

    Parameters
    ----------
    sta : array_like
        The full 3-dimensional STA to be decomposed

    Returns
    -------
    s : array_like
        The spatial kernel

    t : array_like
        The temporal kernel

    """
    _, u, _, v = lowranksta(sta, k=1)
    return v[0].reshape(sta.shape[1:]), u[:, 0]


def _gaussian_function_2d(x, x0, y0, a, b, c):
    """
    A 2D gaussian function

    Parameters
    ----------
    x : array_like
        A (2 by N) array of N data points

    x0 : float
        The x center

    y0 : float
        The y center

    a : float
        The upper left number in the precision matrix

    b : float
        The upper right / lower left number in the precision matrix

    c : float
        The lower right number in the precision matrix

    """

    # center the data
    xn = x[0, :] - x0
    yn = x[1, :] - y0

    # gaussian function
    return np.exp(-0.5*(a*xn**2 + 2*b*xn*yn + c*yn**2))


def _popt_to_ellipse(x0, y0, a, b, c):
    """
    Converts the parameters for the 2D gaussian function (see `fgauss`) into
    ellipse parameters
    """

    # convert precision matrix parameters to ellipse parameters
    u, v = np.linalg.eigh(np.array([[a, b], [b, c]]))

    # standard deviations
    sigmas = np.sqrt(1/u)

    # rotation angle
    theta = np.rad2deg(np.arccos(v[1, 1]))

    return (x0, y0), sigmas, theta


def _smooth_spatial_profile(f, spatial_smoothing, tvd_penalty):
    """
    Smooths a 2D spatial RF profile using a gaussian filter and total variation
    denoising

    Parameters
    ----------
    f : array_like
        2D profile to smooth

    spatial_smoothing : float
        width of the gaussian filter

    tvd_penalty : float
        TV penalty strength (note: larger values indicate a weaker penalty)

    Notes
    -----
    Raises a ValueError if the RF profile is too noisy

    """

    sgn = np.sign(skew(f.ravel()))
    if sgn*skew(f.ravel()) < 0.1:
        raise ValueError("Error! RF profile is too noisy!")

    H = denoise_tv_bregman(gaussian_filter(sgn * f, spatial_smoothing),
                           tvd_penalty)
    return H / H.max()


def _initial_gaussian_params(sta_frame, xm, ym):
    """
    Guesses the initial 2D Gaussian parameters
    """

    # normalize
    wn = (sta_frame / np.sum(sta_frame)).ravel()

    # estimate means
    xc = np.sum(wn * xm.ravel())
    yc = np.sum(wn * ym.ravel())

    # estimate covariance
    data = np.vstack(((xm.ravel() - xc), (ym.ravel() - yc)))
    Q = data.dot(np.diag(wn).dot(data.T)) / (1 - np.sum(wn**2))

    # compute precision matrix
    P = np.linalg.inv(Q)
    a = P[0, 0]
    b = P[0, 1]
    c = P[1, 1]

    return xc, yc, a, b, c


def get_ellipse_params(tx, ty, sta_frame, spatial_smoothing=1.5, tvd_penalty=100):
    """
    Fit an ellipse to the given spatial receptive field and return parameters

    Parameters
    ----------
    sta_frame : array_like
        The spatial receptive field to which the ellipse should be fit

    spatial_smoothing : float, optional

    tvd_penalty : float, optional

    Returns
    -------
    center : (float,float)
        The receptive field center (location stored as an (x,y) tuple)

    widths : [float,float]
        Two-element list of the size of each principal axis of the RF ellipse

    theta : float
        angle of rotation of the ellipse from the vertical axis, in radians

    """

    # preprocess
    ydata = _smooth_spatial_profile(sta_frame, spatial_smoothing, tvd_penalty)

    # get initial params
    xm, ym = np.meshgrid(tx, ty)
    pinit = _initial_gaussian_params(ydata**2, xm, ym)

    # optimize
    xdata = np.vstack((xm.ravel(), ym.ravel()))
    popt, pcov = curve_fit(_gaussian_function_2d, xdata, ydata.ravel(),
                           p0=pinit)

    # return ellipse parameters
    return _popt_to_ellipse(*popt)


def fit_ellipse(tx, ty, sta_frame, spatial_smoothing=1.5, tvd_penalty=100, scale=1.5, **kwargs):
    """
    Fit an ellipse to the given spatial receptive field

    Parameters
    ----------
    sta_frame : array_like
        The spatial receptive field to which the ellipse should be fit

    scale : float, optional
        Scale factor for the ellipse (Default: 1.0)

    Returns
    -------
    ell:
        A matplotlib.patches.Ellipse object

    """

    # Get ellipse parameters
    center, widths, theta = get_ellipse_params(tx, ty, sta_frame,
                                               spatial_smoothing=spatial_smoothing,
                                               tvd_penalty=tvd_penalty)

    # Generate ellipse
    ell = Ellipse(xy=center, width=scale * widths[0],
                  height=scale * widths[1], angle=theta, **kwargs)
    return ell


def filterpeak(sta):
    """
    Find the peak (single point in space/time) of a smoothed filter

    Parameters
    ----------
    sta : array_like
        Filter of which to find the peak (time, space, space)

    Returns
    -------
    idx : int
        Linear index of the maximal point

    sidx : int
        Spatial index of the maximal point

    tidx : int
        Temporal index of the maximal point

    """

    # Find the index of the maximal point
    idx = np.unravel_index(np.abs(sta).argmax(), sta.shape)

    # Split into spatial/temporal indices
    sidx = np.roll(idx[1:], 1)
    tidx = idx[0]

    # Return the indices
    return idx, sidx, tidx


def smoothfilter(f, spacesig=0.5, timesig=1):
    """

    Smooths a 3D spatiotemporal linear filter using a multi-dimensional
    Gaussian filter with the given properties.

    Parameters
    ----------

    f : array_like
        3D filter to be smoothed

    spacesig : float
        The standard deviation of the spatial Gaussian smoothing kernel

    timesig : float
        The standard deviation of the temporal Gaussian smoothing kernel

    Returns
    -------

    fsmooth : array_like
        The smoothed filter, with the same shape as the input

    """
    return ndimage.filters.gaussian_filter(f, (timesig, spacesig, spacesig),
                                           order=0)


def cutout(arr, idx=None, width=5):
    """
    Cut out a chunk of the given stimulus or filter

    Parameters
    ----------
    arr : array_like
        Stimulus or filter array from which the chunk is cut out. The array
        should be shaped as (time, spatial, spatial)

    idx : array_like
        2D array specifying the row and column indices of the center of the
        section to be cut out (if None, the indices are taken from filterpeak)

    width : int
        The size of the chunk to cut out from the start indices

    Returns
    -------
    cut : array_like
        The cut out section of the given stimulus or filter

    """

    if idx is None:
        idx = np.roll(filterpeak(arr)[1], 1)

    # Check idx is a 2-elem array-like
    if len(idx) != 2:
        raise ValueError('idx must be a 2-element array')

    # Find the indices
    row = np.arange(idx[0] - width, idx[0] + width + 1)
    col = np.arange(idx[1] - width, idx[1] + width + 1)

    # Make sure the indices are within the bounds of the given array
    row = row[(row >= 0) & (row < arr.shape[1])]
    col = col[(col >= 0) & (col < arr.shape[2])]

    # Mesh the indices
    rmesh, cmesh = np.meshgrid(row, col)

    # Extract and return the reduced array
    return arr[:, rmesh, cmesh]


def prinangles(u, v):
    """
    Compute the principal angles between two subspaces. Useful for comparing
    subspaces returned via spike-triggered covariance, for example.

    Parameters
    ----------
    u, v : array_like
        The subspaces to compare. They should be of the same size.

    Returns
    -------
    ang : array_like
        The angles between each dimension of the subspaces

    mag : array_like
        The magnitude of the overlap between each dimension of the subspace.

    """

    # Orthogonalize each subspace
    qu, qv = np.linalg.qr(u)[0], np.linalg.qr(v)[0]

    # singular values of the inner product between the orthogonalized spaces
    mag = np.linalg.svd(qu.T.dot(qv), compute_uv=False, full_matrices=False)

    # Compute the angles between each dimension
    ang = np.rad2deg(np.arccos(mag))

    return ang, mag


def rolling_window(array, window, time_axis=-1):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to
    window : int
        Size of rolling window

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

    if time_axis==0:
        array = array.T
    elif time_axis==-1:
        pass
    else:
        raise ValueError('Time axis must be first or last')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    if time_axis==0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr
