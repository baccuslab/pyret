"""
Tools ansd utilities for computing spike-triggered averages (filters),
finding spatial and temporal components of
spatiotemporal filters, and basic filter signal processing.

"""

import numpy as np
import scipy
from numpy.linalg import LinAlgError
from skimage.measure import label, regionprops, find_contours
from functools import reduce

__all__ = ['getste', 'getsta', 'getstc', 'lowranksta', 'decompose',
           'filterpeak', 'smooth', 'cutout', 'rolling_window', 'resample',
           'get_ellipse', 'get_contours', 'get_regionprops',
           'normalize_spatial']


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

    # get the iterator
    ste = getste(time, stimulus, spikes, filter_length)

    # time axis
    tax = time[:filter_length] - time[0]

    # reduce
    try:
        first = next(ste)  # check for empty generators
        sta = reduce(lambda sta, x: np.add(sta, x),
                     ste, first) / float(len(spikes))
    except StopIteration:
        return (np.nan * np.ones((filter_length,) + stimulus.shape[1:]), tax)

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

    # get the blas function for computing the outer product
    outer = scipy.linalg.blas.get_blas_funcs('syr', dtype='d')

    # get the iterator
    ste = getste(time, stimulus, spikes, filter_length)

    # check if empty
    first = next(ste, None)

    # if the spike-triggered ensemble is empty, return an array of NaN's
    if first is None:
        ndims = np.prod(stimulus.shape[1:]) * filter_length
        return np.nan * np.ones((ndims, ndims))

    # initialize the STC matrix using the outer product of the first sample
    stc_init = np.triu(np.outer(first.ravel(), first.ravel()))

    # reduce the stc using the BLAS outer product function
    # (note: this only fills in the upper triangular part of the matrix)
    stc_ut = reduce(lambda C, x: outer(1, x.ravel(), a=C), ste, stc_init)

    # normalize by the number of spikes
    stc_ut /= float(len(spikes))

    # compute the STA (to remove it)
    sta = getsta(time, stimulus, spikes, filter_length)[0].ravel()

    # fill in the lower triangular portion (by adding the transpose)
    # and subtract off the STA to compute the full STC matrix
    stc = np.triu(stc_ut, 1).T + stc_ut - np.outer(sta, sta)

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


def smooth(f, spacesig=0.5, timesig=1):
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
    return scipy.ndimage.filters.gaussian_filter(f,
                                                 (timesig, spacesig, spacesig),
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


def resample(arr, scale_factor):
    """
    Resamples a 1-D or 2-D array
    """

    assert type(arr) is np.ndarray, "Input array must be a numpy array"
    assert scale_factor > 0, "Scale factor must be non-negative"

    if arr.ndim == 1:
        return scipy.signal.resample(arr, int(np.ceil(scale_factor * arr.size)))

    elif arr.ndim == 2:
        assert arr.shape[0] == arr.shape[1], "Array must be square"
        n = int(np.ceil(scale_factor * arr.shape[0]))
        return scipy.signal.resample(scipy.signal.resample(arr, n, axis=0), n, axis=1)

    else:
        raise ValueError('Input array must be either 1-D or 2-D')


def rolling_window(*args, **kwargs):
    """Raise DeprecationWarning until next pyret release (0.4.1)"""
    raise DeprecationWarning('The filtertools.rolling_window function has been moved.\nPlease use stimulustools.rolling_window instead!')


def normalize_spatial(spatial_filter, scale_factor=1.0, clip_negative=False):
    """
    Normalizes a spatial frame by doing the following:
    1. mean subtraction using a robust estimate of the mean (ignoring outliers)
    2. sign adjustment so it is always an 'on' feature
    3. scaling such that the std. dev. of the pixel values is 1.0

    Parameters
    ----------
    spatial_filter : array_like

    scale_factor : float, optional
        The given filter is resampled at a sampling rate of this ratio times
        the original sampling rate (Default: 1.0)

    clip_negative : boolean, optional
        Whether or not to clip negative values to 0. (Default: True)

    """

    # work with a copy of the given filter
    rf = spatial_filter.copy()
    rf -= rf.mean()

    # compute the mean of pixels within +/- 5 std. deviations of the mean
    outlier_threshold = 5 * np.std(rf.ravel())
    mu = rf[(rf <= outlier_threshold) & (rf >= -outlier_threshold)].mean()

    # remove this mean and multiply by the sign of the skew (which forces
    # the polarity of the filter to be an 'ON' feature)
    rf_centered = (rf - mu) * np.sign(scipy.stats.skew(rf.ravel()))

    # normalize by the standard deviation of the pixel values
    rf_centered /= rf_centered.std()

    # resample by the given amount
    rf_resampled = resample(rf_centered, scale_factor)

    # clip negative values
    if clip_negative:
        rf_resampled = np.maximum(rf_resampled, 0)

    return rf_resampled


def get_contours(spatial_filter, threshold=10.0):
    """
    Gets contours of a 2D spatial filter

    Usage
    -----
    >>> rr, cc = get_contours(sta_spatial)
    >>> plt.plot(rr, cc)

    Parameters
    ----------
    spatial_filter : array_like
        The spatial receptive field to which the ellipse should be fit

    threshold : float, optional
        Threshold value (Default: 10.0)

    Returns
    -------
    rr, cc : array_like
        List of contour indices

    """
    return find_contours(normalize_spatial(spatial_filter), threshold)


def get_regionprops(spatial_filter, threshold=10.0):
    """
    Gets region properties of a 2D spatial filter

    Usage
    -----
    >>> regions = get_regionprops(sta_spatial)
    >>> print(regions[0].area) # prints the area of the first region

    Parameters
    ----------
    spatial_filter : array_like
        The spatial receptive field to which the ellipse should be fit

    threshold : float, optional
        Threshold value (Default: 10.0)

    Returns
    -------
    regions : list
        List of region properties (see scikit-image regionprops for more
        information)

    """
    return regionprops(label(normalize_spatial(spatial_filter) >= threshold))


def get_ellipse(spatial_filter, pvalue=0.6827):
    """
    Get the parameters of an ellipse fit to a spatial receptive field

    Parameters
    ----------
    spatial_filter : array_like
        The spatial receptive field to which the ellipse should be fit

    pvalue : float, optional
        Determines the threshold of the ellipse contours. For example, a pvalue
        of 0.95 corresponds to a 95% confidence ellipse. (Default: 0.6827)

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
    zdata = normalize_spatial(spatial_filter, clip_negative=True).ravel()
    zdata /= zdata.max()

    # get initial parameters
    nx, ny = spatial_filter.shape
    xm, ym = np.meshgrid(np.arange(nx), np.arange(ny))
    pinit = _initial_gaussian_params(xm, ym, zdata)

    # optimize
    data = np.vstack((xm.ravel(), ym.ravel()))
    popt, pcov = scipy.optimize.curve_fit(_gaussian_function,
                                          data,
                                          zdata,
                                          p0=pinit)

    # return ellipse parameters, scaled by the appropriate scale factor
    scale = 2 * np.sqrt(scipy.stats.chi2.ppf(pvalue, df=2))
    return _popt_to_ellipse(*popt, scale=scale)


def rfsize(spatial_filter, dx, dy=None, pvalue=0.6827):
    """
    Computes the lenghts of an ellipse fit to the receptive field

    Parameters
    ----------

    spatial_filter : array_like
        The spatial receptive field to which the ellipse should be fit

    dx : float
        The spatial sampling along the x-dimension

    dy : float
        The spatial sampling along the y-dimension. If None, uses the same
        value as dx. (Default: None)

    pvalue : float, optional
        Determines the threshold of the ellipse contours. For example, a pvalue
        of 0.95 corresponds to a 95% confidence ellipse. (Default: 0.6827)

    """

    if dy is None:
        dy = dx

    # x- and y- sampling locations
    tx = np.arange(spatial_filter.shape[0])
    ty = np.arange(spatial_filter.shape[1])

    # get ellipse parameters
    widths = get_ellipse(tx, ty, spatial_filter, pvalue=pvalue)[1]

    # return the scaled widths
    return widths[0] * dx, widths[1] * dy


def _gaussian_function(data, x0, y0, a, b, c):
    """
    A 2D gaussian function (used for fitting an ellipse to RFs)

    Parameters
    ----------
    data : array_like
        A (2 by N) array of N data points

    x0 : float
        The x-location of the center of the ellipse

    y0 : float
        The y-location of the center of the ellipse

    a : float
        The upper left number in the precision matrix

    b : float
        The upper right / lower left number in the precision matrix

    c : float
        The lower right number in the precision matrix

    Returns
    -------
    z : array_like
        The (unnormalized) values of the 2D gaussian function with the given
        parameters

    """

    # center the data
    xc = data[0] - x0
    yc = data[1] - y0

    # gaussian function
    return np.exp(-0.5 * (a * xc**2 + 2 * b * xc * yc + c * yc**2))


def _popt_to_ellipse(y0, x0, a, b, c, scale=3.0):
    """
    Converts the parameters (center and terms in the precision matrix) for a 2D
    gaussian function into ellipse parameters (center, widths, and rotation)

    Parameters
    ----------
    x0 : float
        The x-location of the center of the ellipse

    y0 : float
        The y-location of the center of the ellipse

    a : float
        The upper left number in the precision matrix

    b : float
        The upper right / lower left number in the precision matrix

    c : float
        The lower right number in the precision matrix

    Returns
    -------
    (x0, y0) : tuple
        A tuple containing the center of the ellipse

    sigmas : tuple
        A tuple containing the length of each principal axis of the ellipse

    theta : float
        The angle of rotation of the ellipse, in degrees

    """

    # convert precision matrix parameters to ellipse parameters
    u, v = np.linalg.eigh(np.array([[a, b], [b, c]]))

    # convert precision standard deviations
    sigmas = scale * np.sqrt(1 / u)

    # rotation angle
    theta = np.rad2deg(np.arccos(v[1, 1]))

    return (x0, y0), sigmas, theta


def _initial_gaussian_params(xm, ym, z):
    """
    Guesses the initial 2D Gaussian parameters given a spatial filter
    """

    # normalize
    zn = (z / np.sum(z)).ravel()

    # estimate means
    xc = np.sum(zn * xm.ravel())
    yc = np.sum(zn * ym.ravel())

    # estimate covariance
    data = np.vstack(((xm.ravel() - xc), (ym.ravel() - yc)))
    Q = data.dot(np.diag(zn).dot(data.T)) / (1 - np.sum(zn ** 2))

    # compute precision matrix
    P = np.linalg.inv(Q)
    a = P[0, 0]
    b = P[0, 1]
    c = P[1, 1]

    return xc, yc, a, b, c
