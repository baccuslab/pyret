"""
Tools ansd utilities for computing spike-triggered averages (filters), finding spatial and temporal components of
spatiotemporal filters, and basic filter signal processing.

"""

import numpy as _np
from matplotlib.patches import Ellipse as _Ellipse
from numpy.linalg import LinAlgError
from scipy.ndimage.filters import gaussian_filter as _gaussian_filter
from scipy.linalg.blas import get_blas_funcs
from stimulustools import getcov as _getcov

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
    fk = (u[:,:k].dot(_np.diag(s[:k]).dot(v[:k,:]))).reshape(f.shape)

    # make sure the temporal kernels have the correct sign

    # get out the temporal filter at the RF center
    peakidx = filterpeak(f)[1]
    tsta = f[peakidx[1], peakidx[0], :].reshape(-1,1)
    tsta -= _np.mean(tsta)

    # project onto the temporal filters and keep the sign
    signs = _np.sign((v - _np.mean(v,axis=1)).dot(tsta))

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


def _fit_two_dim_gaussian(histogram, num_samples=10000):
    """
    Fit a 2D gaussian to an empirical histogram

    Parameters
    ----------
    histogram : array_like
        The binned 2D histogram of values

    num_samples : int, optional
        Number of samples to draw when estimating the Gaussian parameters (Default: 10,000)

    """

    # Indices
    x  = _np.linspace(0,1,histogram.shape[0])
    y  = _np.linspace(0,1,histogram.shape[1])
    xx, yy = _np.meshgrid(x, y)

    # Draw samples
    # noinspection PyTypeChecker
    indices = _np.random.choice(_np.flatnonzero(histogram + 1), size=int(num_samples),
                                replace=True, p=histogram.ravel())
    x_samples = xx.ravel()[indices]
    y_samples = yy.ravel()[indices]

    # Fit mean / covariance
    samples = _np.array((x_samples,y_samples))
    center_index = _np.unravel_index(_np.argmax(histogram), histogram.shape)
    center = (xx[center_index], yy[center_index])
    C = _np.cov(samples)

    # Get width / angles
    widths,vectors = _np.linalg.eig(C)
    angle = _np.arccos(vectors[0,0])

    return center, widths, angle


def _image_to_hist(data, spatial_smoothing=2.5):
    """
    Converts 2D image to histogram

    """

    # Smooth the data
    data_smooth = _gaussian_filter(data, spatial_smoothing, order=0)

    # Mean subtract
    mu              = _np.median(data_smooth)
    data_centered   = data_smooth - mu

    # Figure out if it is an on or off profile
    if _np.abs(_np.max(data_centered)) < _np.abs(_np.min(data_centered)):

        # flip from 'off' to 'on'
        data_centered *= -1

    # Min-subtract
    data_centered -= _np.min(data_centered)

    # Normalize to a PDF
    pdf = data_centered / _np.sum(data_centered)

    return pdf


def get_ellipse_params(sta_frame):
    """
    Fit an ellipse to the given spatial receptive field, return parameters of the fit ellipse

    Parameters
    ----------
    sta_frame : array_like
        The spatial receptive field to which the ellipse should be fit

    scale : float
        Scale factor for the ellipse

    Returns
    -------
    center : (float,float)
        The receptive field center (location stored as an (x,y) tuple)

    widths : [float,float]
        Two-element list of the size of each principal axis of the RF ellipse

    theta : float
        angle of rotation of the ellipse from the vertical axis, in radians

    """

    # Get ellipse parameters
    histogram = _image_to_hist(sta_frame)
    return _fit_two_dim_gaussian(histogram)


def fit_ellipse(sta_frame, scale=1.0):
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
    center, widths, theta = get_ellipse_params(sta_frame)

    # Generate ellipse
    ell = _Ellipse(xy=center, width=scale * widths[0], height=scale * widths[1], angle=_np.rad2deg(theta)+45)

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
    -----

    f:
        3D filter to be smoothed

    spacesig:
        The standard deviation of the spatial Gaussian smoothing kernel

    timesig:
        The standard deviation of the temporal Gaussian smoothing kernel

    Returns
    ------

    fsmooth:
        The smoothed filter, with the same shape as the input

    """
    return _gaussian_filter(f, (spacesig, spacesig, timesig), order=0)


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
