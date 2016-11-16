"""
Tools and utilities for computing spike-triggered averages (filters),
finding spatial and temporal components of
spatiotemporal filters, and basic filter signal processing.

"""

import numpy as np
import scipy
from skimage.measure import label, regionprops, find_contours
from functools import reduce

from pyret.stimulustools import slicestim
from pyret.utils import flat2d

__all__ = ['ste', 'sta', 'stc', 'lowranksta', 'decompose',
           'filterpeak', 'smooth', 'cutout', 'resample', 'flat2d',
           'get_ellipse', 'get_contours', 'get_regionprops',
           'normalize_spatial', 'linear_prediction', 'revcorr']


def ste(time, stimulus, spikes, filter_length):
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


def sta(time, stimulus, spikes, filter_length):
    """
    Compute a spike-triggered average

    sta, tax = sta(time, stimulus, spikes, filter_length)

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
    ste_it = ste(time, stimulus, spikes, filter_length)

    # time axis
    tax = time[:filter_length] - time[0]

    # reduce
    try:
        first = next(ste_it)  # check for empty generators
        sta = reduce(lambda sta, x: np.add(sta, x),
                     ste_it, first) / float(len(spikes))
    except StopIteration:
        return (np.nan * np.ones((filter_length,) + stimulus.shape[1:]), tax)

    return sta, tax


def stc(time, stimulus, spikes, filter_length):
    """
    Compute the spike-triggered covariance

    stc = stc(time, stimulus, spikes, filter_length)

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
    ste_it = ste(time, stimulus, spikes, filter_length)

    # check if empty
    first = next(ste_it, None)

    # if the spike-triggered ensemble is empty, return an array of NaN's
    if first is None:
        ndims = np.prod(stimulus.shape[1:]) * filter_length
        return np.nan * np.ones((ndims, ndims))

    # initialize the STC matrix using the outer product of the first sample
    stc_init = np.triu(np.outer(first.ravel(), first.ravel()))

    # reduce the stc using the BLAS outer product function
    # (note: this only fills in the upper triangular part of the matrix)
    stc_ut = reduce(lambda C, x: outer(1, x.ravel(), a=C), ste_it, stc_init)

    # normalize by the number of spikes
    stc_ut /= float(len(spikes))

    # compute the STA (to remove it)
    s = sta(time, stimulus, spikes, filter_length)[0].ravel()

    # fill in the lower triangular portion (by adding the transpose)
    # and subtract off the STA to compute the full STC matrix
    stc = np.triu(stc_ut, 1).T + stc_ut - np.outer(s, s)

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
        the top ``k`` temporal components  (each column is a component).

    s : array_like
        the top ``k`` singular values.

    v : array_like
        the top ``k`` spatial components (each row is a component). These
        components have all spatial dimensions collapsed to one.

    Notes
    -----

    This method requires that the linear filter be 3D. To decompose a
    linear filter into a temporal and 1-dimensional spatial filter, simply
    promote the filter to 3D before calling this method.

    """

    # work with a copy of the filter (prevents corrupting the input)
    f = f_orig.copy() - f_orig.mean()

    # Compute the SVD of the full filter
    assert f.ndim >= 2, "Filter must be at least 2-D"
    u, s, v = np.linalg.svd(f.reshape(f.shape[0], -1), full_matrices=False)

    # Keep the top k components
    k = np.min([k, s.size])
    u = u[:, :k]
    s = s[:k]
    v = v[:k, :]

    # Compute the rank-k filter
    fk = (u.dot(np.diag(s).dot(v))).reshape(f.shape)

    # Ensure that the computed filter components have the correct sign.
    # The full STA should have positive projection onto first temporal
    # component of the low-rank STA.
    sign = np.sign(np.einsum('i,ijk->jk', u[:, 0], f).sum())
    u *= sign
    v *= sign

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
    Find the peak (single point in space/time) of a smoothed filter.

    Parameters
    ----------
    sta : array_like
        Filter of which to find the peak (time, ...), where ellipses
        indicate any spatial dimensions to the stimulus.

    Returns
    -------
    linear_index : int
        Linear index of the maximal point, i.e., treating the array as
        flattened.

    sidx : 1- or 2-element tuple
        Spatial index of the maximal point. This returns a tuple with the
        same number of elements as the filter has spatial dimensions.

    tidx : int
        Temporal index of the maximal point.

    """

    # Find the index of the maximal point
    linear_index = np.abs(sta).argmax()
    idx = np.unravel_index(linear_index, sta.shape)

    # Split into spatial/temporal indices
    sidx = np.roll(idx[1:], 1)
    tidx = idx[0]

    # Return the indices
    return linear_index, sidx, tidx


def smooth(f, spacesig=0.5, timesig=1):  # pragma: no cover
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
    row = np.arange(idx[0] - width, idx[0] + width)
    col = np.arange(idx[1] - width, idx[1] + width)

    # Make sure the indices are within the bounds of the given array
    row = row[(row >= 0) & (row < arr.shape[1])]
    col = col[(col >= 0) & (col < arr.shape[2])]

    # Mesh the indices
    rmesh, cmesh = np.meshgrid(row, col, indexing='ij')

    # Extract and return the reduced array
    return arr[:, rmesh, cmesh]


def resample(arr, scale_factor):
    """Resamples a 1-D or 2-D array by the given scale.

    Parameters
    ----------

    arr : array_like
        The original array to be resampled.

    scale_factor: int_like
        The factor by which `arr` will be resampled. For example, a
        factor of 2 results in an of twice the size in each dimension,
        with points interpolated between existing points.

    Returns
    -------

    res : array_like
        The resampled array. If ``arr`` has shape (M,N), ``res`` has
        shape ``(scale_factor*M, scale_factor*N)``.

    Raises
    ------
    An AssertionError is raised if the scale factor is <= 0.
    A ValueError is raised if the input array is not 1- or 2-dimensional.
    """

    assert scale_factor > 0, "Scale factor must be non-negative"

    if arr.ndim == 1:
        return scipy.signal.resample(arr, int(np.ceil(scale_factor * arr.size)))

    elif arr.ndim == 2:
        assert arr.shape[0] == arr.shape[1], "Array must be square"
        n = int(np.ceil(scale_factor * arr.shape[0]))
        return scipy.signal.resample(scipy.signal.resample(arr, n, axis=0), n, axis=1)

    else:
        raise ValueError('Input array must be either 1-D or 2-D')


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
        Whether or not to clip negative values to 0. (Default: False)

    Returns
    -------
    rf_resampled : array_like
        The normalized (and potentially resampled) filter frame.
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


def get_contours(spatial_filter, threshold=10.0):  # pragma: no cover
    """
    Gets iso-value contours of a 2D spatial filter.

    This returns a list of arrays of shape (n, 2). Each array in the list
    gives the indices into the spatial filter of one contour, and each
    column of the contour gives the indices along the two dimesions of
    the filter.

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


def get_regionprops(spatial_filter, threshold=10.0):  # pragma: no cover
    """
    Gets region properties of a 2D spatial filter.

    This returns various attributes of the non-zero area of the given
    spatial filter, such as its area, centroid, eccentricity, etc.

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


def get_ellipse(spatial_filter, sigma=2.):
    """
    Get the parameters of an ellipse fit to a spatial receptive field

    Parameters
    ----------
    spatial_filter : array_like
        The spatial receptive field to which the ellipse should be fit

    sigma : float, optional
        Determines the size of the ellipse contour, in units of standard
        deviations. (Default: 2)

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
    zdata = normalize_spatial(spatial_filter, clip_negative=True)
    zdata /= zdata.max()

    # get initial parameters
    nx, ny = spatial_filter.shape
    xm, ym = np.meshgrid(np.arange(nx), np.arange(ny))
    pinit = _initial_gaussian_params(xm, ym, zdata)

    # optimize
    data = np.vstack((xm.ravel(), ym.ravel()))
    popt, pcov = scipy.optimize.curve_fit(_gaussian_function,
                                          data,
                                          zdata.ravel(),
                                          p0=pinit)

    # return ellipse parameters, scaled by the appropriate scale factor
    return _popt_to_ellipse(*popt, sigma=sigma)


def rfsize(spatial_filter, dx, dy=None, sigma=2.):
    """
    Computes the lengths of an ellipse fit to the receptive field

    Parameters
    ----------

    spatial_filter : array_like
        The spatial receptive field to which the ellipse should be fit

    dx : float
        The spatial sampling along the x-dimension

    dy : float
        The spatial sampling along the y-dimension. If None, uses the same
        value as dx. (Default: None)

    sigma : float, optional
        Determines the size of the ellipse contour, in units of standard deviation
        of the fitted gaussian. E.g., 2.0 means a 2 SD ellipse.

    Returns
    -------
    xsize, ysize : float
        The x- and y-sizes of the ellipse fitted to the receptive field, at
        the given sigma.

    """

    if dy is None:
        dy = dx

    # get ellipse parameters
    widths = get_ellipse(spatial_filter, sigma=sigma)[1]

    # return the scaled widths
    return widths[0] * dx, widths[1] * dy


def linear_prediction(filt, stim):
    """
    Compute the predicted linear response of a receptive field to a stimulus.

    Parameters
    ----------
    filt : array_like
        The linear filter whose response is to be computed. The array should
        have shape ``(t, ...)``, where ``t`` is the number of time points in the
        filter and the ellipsis indicates any remaining spatial dimenions.
        The number of dimensions and the sizes of the spatial dimensions
        must match that of ``stim``.

    stim : array_like
        The stimulus to which the predicted response is computed. The array
        should have shape (T,...), where ``T`` is the number of time points
        in the stimulus and the ellipsis indicates any remaining spatial
        dimensions. The number of dimensions and the sizes of the spatial
        dimenions must match that of ``filt``.

    Returns
    -------
    pred : array_like
        The predicted linear response. The shape is ``(T - t + 1,)`` where
        ``T`` is the number of time points in the stimulus, and ``t`` is 
        the number of time points in the filter. This is the valid portion
        of the convolution between the stimulus and filter

    Raises
    ------
    ValueError : If the number of dimensions of ``stim`` and ``filt`` do not
        match, or if the spatial dimensions differ.
    """
    if (filt.ndim != stim.ndim) or (filt.shape[1:] != stim.shape[1:]):
        raise ValueError("The filter and stimulus must have the same " +
                         "number of dimensions and match in size along spatial dimensions")

    slices = slicestim(stim, filt.shape[0])
    return np.einsum('tx,x->t', flat2d(slices), filt.ravel())


def revcorr(response, stimulus, filter_length):
    """
    Compute the reverse-correlation between a stimulus and a response.

    This returns the best-fitting linear filter which predicts the given
    response from the stimulus. It is analogous to the spike-triggered
    average for continuous variables. ``response`` is most often a membrane
    potential.

    Parameters
    ----------
    response : array_like
        A continuous output response correlated with the stimulus. Must
        be one-dimensional.

    stimulus : array_like
        A input stimulus correlated with the ``response``. Must be of shape
        ``(t, ...)``, where ``t`` is the time and ``...`` indicates any spatial dimensions.

    filter_length : int
        The length of the returned filter, in samples of the ``stimulus`` and
        ``response`` arrays.

    Returns
    -------
    filt : array_like
        An array of shape ``(filter_length, ...)`` containing the best-fitting
        linear filter which predicts the response from the stimulus. The ellipses
        indicates spatial dimensions of the filter.

    Raises
    ------
    ValueError : If the ``stimulus`` and ``response`` arrays are of different shapes.

    Notes
    -----
    The ``response`` and ``stimulus`` arrays must share the same sampling
    rate. As the stimulus often has a lower sampling rate, one can use
    ``stimulustools.upsamplestim`` to upsample it.
    """
    if response.ndim > 1:
        raise ValueError("The `response` must be 1-dimensional")
    if response.size != (stimulus.shape[0] - filter_length + 1):
        msg = "`stimulus` must have {:#d} time points (`response.size` + `filter_length`)"
        raise ValueError(msg.format(response.size + filter_length + 1))

    slices = slicestim(stimulus, filter_length)
    recovered = np.einsum('tx,t->x', flat2d(slices), response)
    return recovered.reshape(slices.shape[1:])


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


def _popt_to_ellipse(y0, x0, a, b, c, sigma=2.):
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

    sigma : float
        The standard deviation of the ellipse (controls the overall ellipse scale)
        (e.g. for sigma=2, the ellipse is a 2 standard deviation ellipse)

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
    scale = sigma * np.sqrt(scipy.stats.chi2.ppf(0.6827, df=2))
    scaled_sigmas = scale * np.sqrt(1 / u)

    # rotation angle
    theta = np.rad2deg(np.arccos(v[1, 1]))

    return (x0, y0), scaled_sigmas, theta


def _initial_gaussian_params(xm, ym, z, width=5):
    """
    Guesses the initial 2D Gaussian parameters given a spatial filter.

    Parameters
    ----------
    xm : array_like
        The x-points for the filter.

    ym : array_like
        The y-points for the filter.

    z : array_like
        The actual data the parameters of which are guessed.

    width : float, optional
        The expected 1 s.d. width of the RF, in samples. (Default: 5)

    Returns
    -------

    xc, yc : float
        Estimated center points for the data.

    a, b, c : float
        Upper-left, lower-right, and off-diagonal terms for the estimated
        precision matrix.
    """

    # estimate means
    xi = z.sum(axis=0).argmax()
    yi = z.sum(axis=1).argmax()
    yc = xm[xi, yi]
    xc = ym[xi, yi]

    # compute precision matrix entries
    a = 1/width
    b = 0
    c = 1/width

    return xc, yc, a, b, c
