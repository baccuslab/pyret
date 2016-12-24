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
           'get_ellipse', 'get_regionprops', 'normalize_spatial', 
           'linear_response', 'revcorr']


def ste(time, stimulus, spikes, nsamples_before, nsamples_after=0):
    """
    Constructs an iterator over spike-triggered stimuli.

    Parameters
    ----------
    time : ndarray
        The time array corresponding to the stimulus.

    stimulus : ndarray
        A spatiotemporal or temporal stimulus array, where time is the 
        first dimension.

    spikes : iterable
        A list or ndarray of spike times.

    nsamples_before : int
        Number of samples to include in the STE before the spike.

    nsamples_after : int
        Number of samples to include in the STE after the spike,
        which defaults to 0.

    Returns
    -------
    ste : generator
        A generator that yields samples from the spike-triggered ensemble.

    Notes
    -----
    The spike-triggered ensemble (STE) is the set of all stimuli immediately
    surrounding a spike. If the full stimulus distribution is p(s), the STE
    is p(s | spike).

    """
    nb, na = nsamples_before, nsamples_after

    # Bin spikes
    (hist, bins) = np.histogram(spikes, time)

    # Get indices of non-zero firing, truncating spikes before nsamples_before
    # or after nsamples_after
    slices = (stimulus[(idx - nb):(idx + na), ...].astype('float64')
              for idx in np.where(hist > 0)[0]
              if (idx > nb and (idx + na) < len(stimulus)))

    # return the iterator
    return slices


def sta(time, stimulus, spikes, nsamples_before, nsamples_after=0):
    """
    Compute a spike-triggered average.

    Parameters
    ----------
    time : ndarray
        The time array corresponding to the stimulus

    stimulus : ndarray
        A spatiotemporal or temporal stimulus array
        (where time is the first dimension)

    spikes : iterable
        A list or ndarray of spike times

    nsamples_before : int
        Number of samples to include in the STA before the spike

    nsamples_after : int
        Number of samples to include in the STA after the spike (default: 0)

    Returns
    -------
    sta : ndarray
        The spatiotemporal spike-triggered average. Note that time
        increases with increasing array index, i.e. time of the spike is
        at the index for which ``tax == 0``.

    tax : ndarray
        A time axis corresponding to the STA, giving the time relative
        to the spike.

    Notes
    -----
    The spike-triggered average (STA) is the averaged stimulus feature
    conditioned on the presence of a spike. This is widely-used method
    for estimating a neuron's receptive field, and captures the average
    stimulus feature to which the neuron responds.

    Formally, the STA is defined as the function [1]:

    .. math::
        C(\\tau) = \\frac{1}{N} \\sum_{i=1}^{N} s(t_i - \\tau)

    where :math:`\\tau` is time preceding the spike, and :math:`t_i` is
    the time of the ith spike.

    The STA is often used to estimate a linear filter which captures
    a neuron's responses. If the stimulus is uncorrelated (spherical),
    the STA is unbiased and proportional to the time-reverse of the
    linear filter.

    References
    ----------
    [1] Dayan, P. and L.F. Abbott. Theoretical Neuroscience: Computational
    and Mathematical Modeling of Neural Systems. 2001.

    """

    # get the iterator
    ste_it = ste(time, stimulus, spikes, nsamples_before, nsamples_after=nsamples_after)

    # time axis
    filter_length = nsamples_before + nsamples_after
    dt = np.mean(np.diff(time))
    tax = dt * np.arange(-nsamples_before, nsamples_after) + dt

    # reduce
    try:
        first = next(ste_it)  # check for empty generators
        sta = reduce(lambda sta, x: np.add(sta, x),
                     ste_it, first) / float(len(spikes))
    except StopIteration:
        return (np.nan * np.ones((filter_length,) + stimulus.shape[1:]), tax)

    return sta, tax


def stc(time, stimulus, spikes, nsamples_before, nsamples_after=0):
    """
    Compute the spike-triggered covariance.

    Parameters
    ----------
    time : ndarray
        The time array corresponding to the stimulus, where time is the 
        first dimension.

    stimulus : ndarray
        A spatiotemporal or temporal stimulus array.

    spikes : iterable
        A list or ndarray of spike times.

    nsamples_before : int
        Number of samples to include in the STC before the spike.

    nsamples_after : int
        Number of samples to include in the STC after the spike,
        which defaults to 0.

    Returns
    -------
    stc : ndarray
        The spike-triggered covariance (STC) matrix.

    """
    # get the blas function for computing the outer product
    outer = scipy.linalg.blas.get_blas_funcs('syr', dtype='d')

    # get the iterator
    ste_it = ste(time, stimulus, spikes, nsamples_before, nsamples_after=nsamples_after)

    filter_length = nsamples_before + nsamples_after

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


def lowranksta(sta_orig, k=10):
    """
    Constructs a rank-k approximation to the given spatiotemporal STA.
    This is useful for estimating a spatial and temporal kernel for an
    STA or for denoising.

    Parameters
    ----------
    sta_orig : array_like
        3D STA to be separated, shaped as ``(time, space, space)``.

    k : int
        Number of components to keep (rank of the reduced STA).

    Returns
    -------
    sk : array_like
        The rank-k estimate of the original STA.

    u : array_like
        The top ``k`` temporal components (each column is a component).

    s : array_like
        The top ``k`` singular values.

    v : array_like
        The top ``k`` spatial components (each row is a component). These
        components have all spatial dimensions collapsed to one.

    Notes
    -----
    This method requires that the STA be 3D. To decompose a STA into a 
    temporal and 1-dimensional spatial component, simply promote the STA 
    to 3D before calling this method.

    Despite the name this method accepts both an STA or a linear filter.
    The components estimated for one will be flipped versions of the other.

    """

    # work with a copy of the STA (prevents corrupting the input)
    f = sta_orig.copy() - sta_orig.mean()

    # Compute the SVD of the full STA
    assert f.ndim >= 2, "STA must be at least 2-D"
    u, s, v = np.linalg.svd(f.reshape(f.shape[0], -1), full_matrices=False)

    # Keep the top k components
    k = np.min([k, s.size])
    u = u[:, :k]
    s = s[:k]
    v = v[:k, :]

    # Compute the rank-k STA
    sk = (u.dot(np.diag(s).dot(v))).reshape(f.shape)

    # Ensure that the computed STA components have the correct sign.
    # The full STA should have positive projection onto first temporal
    # component of the low-rank STA.
    sign = np.sign(np.einsum('i,ijk->jk', u[:, 0], f).sum())
    u *= sign
    v *= sign

    # Return the rank-k approximate STA, and the SVD components
    return sk, u, s, v


def decompose(sta):
    """
    Decomposes a spatiotemporal STA into a spatial and temporal kernel.

    Parameters
    ----------
    sta : array_like
        The full 3-dimensional STA to be decomposed, of shape ``(t, nx, ny)``.

    Returns
    -------
    s : array_like
        The spatial kernel, with shape ``(nx * ny,)``.

    t : array_like
        The temporal kernel, with shape ``(t,)``.

    """
    _, u, _, v = lowranksta(sta, k=1)
    return v[0].reshape(sta.shape[1:]), u[:, 0]


def filterpeak(sta):
    """
    Find the peak (single point in space/time) of a smoothed STA or
    linear filter.

    Parameters
    ----------
    sta : array_like
        STA or filter for which to find the peak. It should be shaped as
        ``(time, ...)``, where ellipses indicate any spatial dimensions 
        to the array.

    Returns
    -------
    linear_index : int
        Linear index of the maximal point, i.e. treating the array as
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
    Smooths a 3D spatiotemporal STA or linear filter using a multi-dimensional
    Gaussian filter with the given properties.

    Parameters
    ----------

    f : array_like
        3D STA or filter to be smoothed.

    spacesig : float
        The standard deviation of the spatial Gaussian smoothing kernel.

    timesig : float
        The standard deviation of the temporal Gaussian smoothing kernel.

    Returns
    -------

    fsmooth : array_like
        The smoothed filter, with the same shape as the input.

    """
    return scipy.ndimage.filters.gaussian_filter(f,
                                                 (timesig, spacesig, spacesig),
                                                 order=0)


def cutout(arr, idx=None, width=5):
    """
    Cut out a chunk of the given stimulus or filter.

    Parameters
    ----------
    arr : array_like
        Stimulus, STA, or filter array from which the chunk is cut out. The array
        should be shaped as ``(time, spatial, spatial)``.

    idx : array_like, optional
        2D array specifying the row and column indices of the center of the
        section to be cut out (if None, the indices are taken from ``filterpeak``).

    width : int, optional
        The size of the chunk to cut out from the start indices. Defaults
        to 5 samples.

    Returns
    -------
    cut : array_like
        The cut out section of the given stimulus, STA, or filter.

    Notes
    -----
    This method can be useful to reduce the space and time costs of computations
    involving stimuli and/or filters. For example, a neuron's receptive field is
    often much smaller than the stimulus, but this method can be used to only
    compare the relevant portions of the stimulus and receptive field.

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
    """
    Resamples a 1-D or 2-D array by the given scale.

    Parameters
    ----------

    arr : array_like
        The original array to be resampled.

    scale_factor: int_like
        The factor by which ``arr`` will be resampled. For example, a
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


def normalize_spatial(frame, scale_factor=1.0, clip_negative=False):
    """
    Normalizes a spatial frame, for example of a stimulus or STA, by 
    doing the following:

    1. mean subtraction using a robust estimate of the mean (ignoring outliers).
    2. scaling such that the std. dev. of the pixel values is 1.0.

    Parameters
    ----------
    frame : array_like
        The spatial frame to be normalized.

    scale_factor : float, optional
        The given frame is resampled at a sampling rate of this ratio times
        the original sampling rate (Default: 1.0).

    clip_negative : boolean, optional
        Whether or not to clip negative values to 0. (Default: False).

    Returns
    -------
    resampled : array_like
        The normalized (and potentially resampled) frame.
    """

    # work with a copy of the given frame
    f = frame.copy()
    f -= f.mean()

    # compute the mean of pixels within +/- 5 std. deviations of the mean
    outlier_threshold = 5 * np.std(f.ravel())
    mu = f[(f <= outlier_threshold) & (f >= -outlier_threshold)].mean()

    # normalize by the standard deviation of the pixel values
    f_centered = f - mu
    f_centered /= f_centered.std()

    # resample by the given amount
    f_resampled = resample(f_centered, scale_factor)

    # clip negative values
    if clip_negative:
        f_resampled = np.maximum(f_resampled, 0)

    return f_resampled


def get_regionprops(spatial_filter, percentile=0.95):  # pragma: no cover
    """
    Gets region properties of a 2D spatial STA or linear filter.

    This returns various attributes of the non-zero area of the given
    spatial filter, such as its area, centroid, eccentricity, etc.

    Usage
    -----
    >>> regions = get_regionprops(sta_spatial)
    >>> print(regions[0].area) # prints the area of the first region

    Parameters
    ----------
    spatial_filter : array_like
        The spatial linear filter to which the ellipse should be fit.

    percentile : float, optional
        The cutoff percentile at which the contour is taken. Defaults
        to 0.95.

    Returns
    -------
    regions : list
        List of region properties (see ``skimage.measure.regionprops`` 
        for more information).

    """
    normed = normalize_spatial(spatial_filter)
    threshold = normed.max() * percentile
    return regionprops(label(normed >= threshold))


def get_ellipse(spatial_filter, sigma=2.):
    """
    Get the parameters of an ellipse fit to a spatial STA or linear filter.

    Parameters
    ----------
    spatial_filter : array_like
        The spatial receptive field to which the ellipse should be fit.

    sigma : float, optional
        Determines the size of the ellipse contour, in units of standard
        deviations. (Default: 2)

    Returns
    -------
    center : (float,float)
        The receptive field center (location stored as an (x,y) tuple).

    widths : [float,float]
        Two-element list of the size of each principal axis of the RF ellipse.

    theta : float
        angle of rotation of the ellipse from the vertical axis, in radians.
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
    Computes the lengths of the major and minor axes of an ellipse fit 
    to an STA or linear filter.

    Parameters
    ----------

    spatial_filter : array_like
        The spatial receptive field to which the ellipse should be fit.

    dx : float
        The spatial sampling along the x-dimension.

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


def linear_response(filt, stim, nsamples_after=0):
    """
    Compute the response of a linear filter to a stimulus.

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

    nsamples_after : int, optional
        The number of acausal points in the filter. Defaults to 0.

    Returns
    -------
    pred : array_like
        The predicted linear response. The shape is ``(T - t + 1,)`` where
        ``T`` is the number of time points in the stimulus, and ``t`` is 
        the number of time points in the filter. This is the valid portion
        of the convolution between the stimulus and filter.

    Raises
    ------
    ValueError : If the number of dimensions of ``stim`` and ``filt`` do not
        match, or if the spatial dimensions differ.

    Notes
    -----
    Both ``filtertools.sta`` and ``filtertools.revcorr`` can estimate "acausal"
    components, such as points in the stimulus occuring *after* a spike. The
    value passed as parameter ``nsamples_after`` must match that value used
    when calling ``filtertools.sta`` or ``filtertools.revcorr``.

    """
    if (filt.ndim != stim.ndim) or (filt.shape[1:] != stim.shape[1:]):
        raise ValueError("The filter and stimulus must have the same " +
                         "number of dimensions and match in size along spatial dimensions")

    slices = slicestim(stim, filt.shape[0] - nsamples_after, nsamples_after)
    return np.einsum('tx,x->t', flat2d(slices), filt.ravel())


def revcorr(stimulus, response, nsamples_before, nsamples_after=0):
    """
    Compute the reverse-correlation between a stimulus and a response.

    Parameters
    ----------
    stimulus : array_like
        A input stimulus correlated with the ``response``. Must be of shape
        ``(t, ...)``, where ``t`` is the time and ``...`` indicates any spatial 
        dimensions.

    response : array_like
        A continuous output response correlated with ``stimulus``. Must
        be one-dimensional, of size ``t``.

    nsamples_before : int
        The maximum negative lag for the correlation between stimulus and response,
        in samples.

    nsamples_after : int, optional
        The maximum positive lag for the correlation between stimulus and response,
        in samples. Defaults to 0.

    Returns
    -------
    rc : array_like
        An array of shape ``(nsamples_before + nsamples_after, ...)``
        containing the best-fitting linear filter which predicts the response from
        the stimulus. The ellipses indicates spatial dimensions of the filter.

    lags : array_like
        An array of shape ``(nsamples_before + nsamples_after,)``, which gives
        the lags, in samples, between ``stimulus`` and ``response`` for the correlation
        returned in ``rc``. This can be converted to an axis of time (like that 
        returned from ``filtertools.sta``) by multiplying by the sampling period.

    Raises
    ------
    ValueError : If the ``stimulus`` and ``response`` arrays are of different shapes.

    Notes
    -----
    The ``response`` and ``stimulus`` arrays must share the same sampling
    rate. As the stimulus often has a lower sampling rate, one can use
    ``stimulustools.upsamplestim`` to upsample it.

    Reverse correlation is a method analogous to spike-triggered averaging for
    continuous response variables, such as a membrane voltage recording. It 
    estimates the stimulus feature that most strongly correlates with the 
    response on average.

    It is the time-reverse of the standard cross-correlation function, and is defined
    as:

    .. math::
        c[-k] = \\sum_{n} s[n] r[n - k]

    The parameter ``k`` is the lag between the two signals in samples. The range
    of lags computed in this method are determined by ``nsamples_before`` and 
    ``nsamples_after``.

    Note that, as with ``filtertools.sta``, the values (samples) in the ``lags``
    array increase with increasing array index. This means that time is moving
    forward with increasing array index.

    """
    history = nsamples_before + nsamples_after
    if response.ndim > 1:
        raise ValueError("The `response` must be 1-dimensional")
    if response.size != (stimulus.shape[0] - history + 1):
        msg = ('`stimulus` must have {:#d} time points ' + 
                '(`response.size` + `nsamples_before` + `nsamples_after`)')
        raise ValueError(msg.format(response.size + history + 1))

    slices = slicestim(stimulus, nsamples_before, nsamples_after)
    recovered = np.einsum('tx,t->x', flat2d(slices), response).reshape(slices.shape[1:])
    lags = np.arange(-nsamples_before, nsamples_after)
    return recovered, lags


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
