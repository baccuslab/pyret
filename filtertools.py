"""
Tools ansd utilities for computing spike-triggered averages (filters), finding spatial and temporal components of
spatiotemporal filters, and basic filter signal processing.

"""

import numpy as _np
from matplotlib.patches import Ellipse as _Ellipse
from numpy.linalg import LinAlgError
from scipy.linalg.blas import get_blas_funcs
from scipy import ndimage as _ndimage
from stimulustools import getcov as _getcov
from scipy.stats import skew
from skimage.restoration import denoise_tv_bregman
from skimage.filter import gaussian_filter
from scipy.optimize import curve_fit


def getste(time, stimulus, spikes, filter_length, tproj=None):
    """
    Construct the spike-triggered ensemble

    Parameters
    ----------
    time : array_like
        The time axis of the stimulus

    stimulus : array_like
        The stimulus array. The last dimension of the stimulus
        array is assumed to be time, but no other restrictions
        are placed on its shape. It works for purely temporal
        and spatiotemporal stimuli.

    spikes : array_like
        Array of spike times

    filter_length : int
        Number of frames over which to construct the ensemble.

    tproj : array_like
        A basis onto which the raw ensemble is projected. This is
        useful for smoothing or reducing the size/dimensionality of
        the ensemble. If None (default), uses the identity matrix.

    Returns
    -------
    ste : array_like
        The spike-triggered stimulus ensemble. The returned array is
        reshaped from the input `stimulus` array, such that all spatial
        dimensions are collapsed. The array has shape
        (nspikes, n_spatial_dims, filterlength).

    steproj : array_like
        The spike-triggered stimulus ensemble, projected onto the
        basis defined by `tproj`. If `tproj` is None (the default), the
        return value here is None.

    tax : array_like
        The time axis of the ensemble. It is of length `filterlength`,
        with intervals given by the sample rate of the `time` input
        array.

    Raises
    ------
    A ValueError is raised if there are no spikes within the requested `time`

    """

    # Bin spikes
    (hist, bins) = _np.histogram(spikes, time)

    # Get indices of non-zero firing, truncating spikes earlier
    # than `filterlength` frames
    nzhist = _np.where(hist > 0)[0]
    nzhist = nzhist[nzhist > filter_length]

    # Collapse any spatial dimensions of the stimulus array
    cstim = stimulus.reshape(-1, stimulus.shape[-1])

    # Preallocate STE array
    ste = _np.empty((nzhist.size, cstim.shape[0], filter_length))

    # Compute the STE, and optionally the projection onto tproj
    if tproj is not None:

        # Preallocate the projection array
        steproj = _np.empty((nzhist.size, cstim.shape[0], filter_length))

        # Loop over spikes, adding filterlength frames preceding each spike
        for idx, val in enumerate(nzhist):

            # Raw STE
            ste[idx, :, :] = cstim[:, (val - filter_length):val]

            # Projected STE
            steproj[idx, :, :] = ste[idx, :, :].dot(tproj).dot(tproj.T)

    else:

        # Projected STE is None
        steproj = None

        # Loop over spikes, adding filterlength frames preceding each spike
        for idx, val in enumerate(nzhist):

            # Raw STE only
            ste[idx, :, :] = cstim[:, (val - filter_length):val]

    # Construct a time axis to return
    tax = time[:filter_length] - time[0]

    # Return STE and the time axis
    return ste, steproj, tax


def getsta(time, stimulus, spikes, filter_length, norm=True, return_flag=0):
    """
    Compute the spike-triggered average

    Parameters
    ----------
    time : array_like
        The time axis of the stimulus

    stimulus : array_like
        The stimulus array. The last dimension of the stimulus
        array is assumed to be time, but no other restrictions
        are placed on its shape. It works for purely temporal
        and spatiotemporal stimuli.

    spikes : array_like
        Array of spike times.

    filter_length : int
        Number of frames over which to construct the
        ensemble

    norm : boolean
        Normalize the computed filter by mean-subtracting and normalizing
        to a unit vector.

    return_flag : int
        0:  (default) returns both sta and tax
        1:  returns only the sta
        2:  returns only the tax

    Returns
    -------
    sta : array_like
        The spike-triggered average. The returned array has
        stimulus.ndim + 1 dimensions, and has a shape of
        (nspikes, stimulus.shape[:-1], filter_length).

    tax : array_like
        The time axis of the ensemble. It is of length `filter_length`,
        with intervals given by the sample rate of the `time` input
        array.

    Raises
    ------
    If no spikes occurred during the given `time` array, a UserWarning
    is raised, and the returned STA is an array of zeros with the desired
    shape (stimulus.shape[:-1], filter_length). This allows the
    STA to play nicely with later functions using it, for example, adding
    multiple STAs together.

    """

    # Bin spikes
    (hist, bins) = _np.histogram(spikes, time)

    # Get indices of non-zero firing, truncating spikes earlier
    # than `filter_length` frames
    nzhist = _np.where(hist > 0)[0]
    nzhist = nzhist[nzhist > filter_length]

    # Check if there are no spikes during this time
    if not _np.any(nzhist):
        import warnings as _wrn
        _wrn.warn('There are no spikes during the requested time')
        sta = _np.zeros(stimulus.shape[:-1] + (filter_length,))
        tax = time[:filter_length] - time[filter_length - 1]

    else:
        # Collapse any spatial dimensions of the stimulus array
        cstim = stimulus.reshape(-1, stimulus.shape[-1])

        # Preallocate STA array
        sta = _np.zeros((cstim.shape[0], filter_length))

        # Add filter_length frames preceding each spike to the running STA
        for idx in nzhist:
            sta += hist[idx] * cstim[:, (idx - filter_length):idx]

        # Mean-subtract and normalize as a vector
        if norm:
            sta -= _np.mean(sta)
            sta /= _np.linalg.norm(sta)

        # otherwise, normalize by dividing by the number of spikes
        else:
            sta /= _np.sum(hist[nzhist])

        # Construct a time axis to return
        tax = time[:filter_length] - time[filter_length - 1]

        # Reshape the STA and flip the time axis so that the time of the spike is at index 0
        sta = _np.reshape(sta, stimulus.shape[:-1] + (filter_length,))

    # Return STA and the time axis
    if return_flag == 0:
        return sta, tax
    elif return_flag == 1:
        return sta
    elif return_flag == 2:
        return tax
    else:
        raise ValueError('return_flag has to be either 0, 1 or 2 in getsta')


def getstc(time, stimulus, spikes, filterlength, tproj=None):
    """
    Compute the spike-triggered covariance

    Usage: cells, tax, stimcov = getstc(time, stimulus, spikes, filterlength, tproj=None)

    Notes
    -----
    We define the dimensionality `d` to be: d = stimulus.shape[:-1] * filterlength

    Parameters
    ----------
    time : array_like
        The time axis of the stimulus

    stimulus : array_like
        The stimulus array. The last dimension of the stimulus
        array is assumed to be time, but no other restrictions
        are placed on its shape. It works for purely temporal
        and spatiotemporal stimuli.

    spikes : list of array_like
        List of arrays of spike times, one for each cell

    filterlength : int
        Number of frames over which to construct the
        ensemble

    tproj : array_like, optional
        Temporal basis set to use. Must have # of rows (first dimension) equal to filterlength.
        Each extracted stimulus slice is projected onto this basis set, which reduces the size
        of the corresponding covariance matrix to store. This basis can be chosen to be some smooth
        set of tiled functions, such as raised cosines, which enforces smooth filters in time.

    Returns
    -------
    cells : list
        contains the following for each cell

        eigvecs : array_like
            The (d x d) set of eigenvectors of the normalized STC matrix, each column is a separate eigenvector

        eigvals : array_like
            The corresponding set of d eigenvalues of the normalized STC matrix

        spkcov : array_like
            The (d x d) spike-triggered covariance matrix.

        sta : array_like
            The spike-triggered average

    stimcov : array_like
        The (d by d) stimulus covariance matrix

    tax : array_like
        The time axis of the ensemble. It is of length `filterlength`,
        with intervals given by the sample rate of the `time` input
        array.

    """

    # temporal basis (if not given, use the identity matrix)
    if tproj is None:
        tproj = _np.eye(filterlength)

    if tproj.shape[0] != filterlength:
        raise ValueError('The first dimension of the basis set tproj must equal filterlength')

    # get the stimulus covariance matrix
    stimcov = _getcov(stimulus, filterlength, tproj=tproj)

    # store information about cells in a list
    cells = list()

    # Construct a time axis to return
    tax = time[:filterlength] - time[0]

    # for each cell's spike times
    for spk in spikes:

        print('[Cell %i of %i]' % (len(cells) + 1, len(spikes)))

        # Bin spikes
        (hist, bins) = _np.histogram(spk, time)

        # Get indices of non-zero firing, truncating spikes earlier
        # than `filterlength` frames
        nzhist = _np.where(hist > 0)[0]
        nzhist = nzhist[nzhist > filterlength]

        # Collapse any spatial dimensions of the stimulus array
        cstim = stimulus.reshape(-1, stimulus.shape[-1])

        # Preallocate STA array and STC matrix
        sta = _np.zeros((cstim.shape[0] * tproj.shape[1], 1))
        spkcov = _np.zeros((cstim.shape[0] * tproj.shape[1], cstim.shape[0] * tproj.shape[1]))

        # get blas function
        blas_ger_fnc = get_blas_funcs(('ger',), (spkcov,))[0]

        # Add the outerproduct of stimulus slices to the STC, keep track of the STA
        for idx in nzhist:

            # get the stimulus slice
            stimslice = (hist[idx] * cstim[:, (idx - filterlength):idx]).dot(tproj).reshape(-1,1)

            # update the spike-triggered average
            sta += stimslice

            # add it to the covariance matrix (using low-level BLAS operation)
            blas_ger_fnc(hist[idx], stimslice, stimslice, a=spkcov.T, overwrite_a=True)

        # normalize and compute the STA outer product
        sta /= float(nzhist.size)
        sta_op = sta.dot(sta.T)

        # mean-subtract and normalize the STC by the number of samples
        spkcov = spkcov / (float(nzhist.size) - 1) - sta_op

        # estimate eigenvalues and eigenvectors of the normalized STC matrix
        try:
            eigvals, eigvecs = _np.linalg.eig(spkcov - stimcov)
        except _np.linalg.LinAlgError:
            print('Warning: eigendecomposition did not converge. You may not have enough data.')
            eigvals = None
            eigvecs = None

        # store results
        cells.append({'sta': sta, 'eigvals': eigvals, 'eigvecs': eigvecs, 'spkcov': spkcov})

    # Return values
    return cells, tax, stimcov


def lowranksta(f_orig, k=10):
    """
    Constructs a rank-k approximation to the given spatiotemporal filter.
    This is useful for computing a spatial and temporal kernel of an STA,
    or for denoising.

    Parameters
    ----------
    f : array_like
        3-D filter to be separated

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
        u, s, v = _np.linalg.svd(f.reshape(-1, f.shape[-1]) - _np.mean(f), full_matrices=False)
    except LinAlgError:
        print('The SVD did not converge for the given spatiotemporal filter')
        print('The data is likely too noisy to compute a rank-{0} approximation'.format(k))
        print('Try reducing the requested rank.')
        return None, None, None, None

    # Keep the top k components
    k = _np.min([k, s.size])

    # Compute the rank-k filter
    fk = (u[:, :k].dot(_np.diag(s[:k]).dot(v[:k, :]))).reshape(f.shape)

    # make sure the temporal kernels have the correct sign

    # get out the temporal filter at the RF center
    peakidx = filterpeak(f)[1]
    tsta = f[peakidx[1], peakidx[0], :].reshape(-1, 1)
    tsta -= _np.mean(tsta)

    # project onto the temporal filters and keep the sign
    signs = _np.sign((v - _np.mean(v, axis=1)).dot(tsta))

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
    return u[:, 0].reshape(sta.shape[:2]), v[0, :]


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
    return _np.exp(-0.5*(a*xn**2 + 2*b*xn*yn + c*yn**2))


def _popt_to_ellipse(x0, y0, a, b, c):
    """
    Converts the parameters for the 2D gaussian function (see `fgauss`) into ellipse parameters
    """

    # convert precision matrix parameters to ellipse parameters
    u, v = _np.linalg.eigh(_np.array([[a, b], [b, c]]))

    # standard deviations
    sigmas = _np.sqrt(1/u)

    # rotation angle
    theta = _np.rad2deg(_np.arccos(v[1, 1]))

    return (x0, y0), sigmas, theta


def _smooth_spatial_profile(f, spatial_smoothing, tvd_penalty):
    """
    Smooths a 2D spatial RF profile using a gaussian filter and total variation denoising

    Parameters
    ----------
    f : array_like
        2D profile to smooth

    spatial_smoothing : float
        width of the gaussian filter

    tvd_penalty : float
        strength of the total variation penalty (note: large values correspond to weaker penalty)

    Notes
    -----
    Raises a ValueError if the RF profile is too noisy

    """

    sgn = _np.sign(skew(f.ravel()))
    if sgn*skew(f.ravel()) < 0.1:
        raise ValueError("Error! RF profile is too noisy!")

    H = denoise_tv_bregman(gaussian_filter(sgn * f, spatial_smoothing), tvd_penalty)
    return H / H.max()


def _initial_gaussian_params(sta_frame, xm, ym):
    """
    Guesses the initial 2D Gaussian parameters
    """

    # normalize
    wn = (sta_frame / _np.sum(sta_frame)).ravel()

    # estimate means
    xc = _np.sum(wn * xm.ravel())
    yc = _np.sum(wn * ym.ravel())

    # estimate covariance
    data = _np.vstack(((xm.ravel() - xc), (ym.ravel() - yc)))
    Q = data.dot(_np.diag(wn).dot(data.T)) / (1 - _np.sum(wn**2))

    # compute precision matrix
    P = _np.linalg.inv(Q)
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
    xm, ym = _np.meshgrid(tx, ty)
    pinit = _initial_gaussian_params(ydata**2, xm, ym)

    # optimize
    xdata = _np.vstack((xm.ravel(), ym.ravel()))
    popt, pcov = curve_fit(_gaussian_function_2d, xdata, ydata.ravel(), p0=pinit)

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
    ell = _Ellipse(xy=center, width=scale * widths[0],
                   height=scale * widths[1], angle=theta, **kwargs)
    return ell


def filterpeak(sta):
    """
    Find the peak (single point in space/time) of a smoothed filter

    Parameters
    ----------
    sta : array_like
        Filter of which to find the peak

    Returns
    -------
    idx : int
        Linear index of the maximal point

    sidx : int
        Spatial index of the maximal point

    tidx : int
        Temporal index of the maximal point

    """

    # Smooth filter
    fs = smoothfilter(sta, spacesig=0.7, timesig=1)

    # Find the index of the maximal point
    idx = _np.unravel_index(_np.abs(fs).argmax(), fs.shape)

    # Split into spatial/temporal indices
    sidx = _np.roll(idx[:2], 1)
    tidx = idx[-1]

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
    return _ndimage.filters.gaussian_filter(f, (spacesig, spacesig, timesig), order=0)


def cutout(arr, idx, width=5):
    """
    Cut out a chunk of the given stimulus or filter

    Parameters
    ----------
    arr : array_like
        Stimulus or filter array from which the chunk is cut out. The array
        should be shaped as (pix, pix, time).

    idx : array_like
        2D array specifying the row and column indices of the center of the
        section to be cut out

    width : int
        The size of the chunk to cut out from the start indices

    Returns
    -------
    cut : array_like
        The cut out section of the given stimulus or filter

    """

    # Check idx is a 2-elem array-like
    if len(idx) != 2:
        raise ValueError('idx must be a 2-element array')

    # Find the indices
    row = _np.arange(idx[0] - width, idx[0] + width + 1)
    col = _np.arange(idx[1] - width, idx[1] + width + 1)

    # Make sure the indices are within the bounds of the given array
    row = row[(row >= 0) & (row < arr.shape[0])]
    col = col[(col >= 0) & (col < arr.shape[1])]

    # Mesh the indices
    rmesh, cmesh = _np.meshgrid(row, col)

    # Extract and return the reduced array
    return arr[rmesh, cmesh, :]


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
    (qu, ru), (qv, rv) = _np.linalg.qr(u), _np.linalg.qr(v)

    # Compute singular values of the inner product between the orthogonalized spaces
    mag = _np.linalg.svd(qu.T.dot(qv), compute_uv=False, full_matrices=False)

    # Compute the angles between each dimension
    ang = _np.rad2deg(_np.arccos(mag))

    return ang, mag
