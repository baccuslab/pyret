'''
filtertools.py

Tools for computation of basic linear filters

(C) 2014 bnaecker, nirum
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.ndimage.filters import gaussian_filter
from stimulustools import getcov

def getste(time, stimulus, spikes, filterlength):
    '''
    
    Construct the spike-triggered ensemble

    Input
    -----

    time (ndarray):
        The time axis of the stimulus

    stimulus (ndarray):
        The stimulus array. The last dimension of the stimulus
        array is assumed to be time, but no other restrictions
        are placed on its shape. It works for purely temporal
        and spatiotemporal stimuli.

    spikes (ndarray):
        Array of spike times.

    filterlength (int):
        Number of frames over which to construct the
        ensemble

    Output
    ------

    ste (ndarray):
        The spike-triggered stimulus ensemble. The returned array
        has stimulus.ndim + 1 dimensions, and has a shape of
        (nspikes, stimulus.shape[:-1], filterlength).

    tax (ndarray):
        The time axis of the ensemble. It is of length `filterlength`,
        with intervals given by the sample rate of the `time` input
        array.

    '''

    # Bin spikes
    (hist, bins) = np.histogram(spikes, time)

    # Get indices of non-zero firing, truncating spikes earlier
    # than `filterlength` frames
    nzhist = np.where(hist > 0)[0]
    nzhist = nzhist[nzhist > filterlength]

    # Collapse any spatial dimensions of the stimulus array
    cstim = stimulus.reshape(-1, stimulus.shape[-1])

    # Preallocate STE array
    ste = np.empty((nzhist.size, cstim.shape[0], filterlength))

    # Add filterlength frames preceding each spike to the STE array
    for idx, val in enumerate(nzhist):
        ste[idx, :, :] = hist[val] * cstim[:, val - filterlength : val]

    # Construct a time axis to return
    tax = time[:filterlength] - time[0]

    # Reshape the STE and flip the time axis so that the time of the spike is at index 0
    ste = np.reshape(ste, (nzhist.size,) + stimulus.shape[:-1] + (filterlength,))
    ste = np.take(ste, np.arange(filterlength - 1, -1, -1), axis=-1)

    # Return STE and the time axis
    return ste, tax

def getsta(time, stimulus, spikes, filterlength):
    '''
    
    Compute the spike-triggered average

    Input
    -----

    time (ndarray):
        The time axis of the stimulus

    stimulus (ndarray):
        The stimulus array. The last dimension of the stimulus
        array is assumed to be time, but no other restrictions
        are placed on its shape. It works for purely temporal
        and spatiotemporal stimuli.

    spikes (ndarray):
        Array of spike times.

    filterlength (int):
        Number of frames over which to construct the
        ensemble

    Output
    ------

    sta (ndarray):
        The spike-triggered average. The returned array has 
        stimulus.ndim + 1 dimensions, and has a shape of
        (nspikes, stimulus.shape[:-1], filterlength).

    tax (ndarray):
        The time axis of the ensemble. It is of length `filterlength`,
        with intervals given by the sample rate of the `time` input
        array.

    '''

    # Bin spikes
    (hist, bins) = np.histogram(spikes, time)

    # Get indices of non-zero firing, truncating spikes earlier
    # than `filterlength` frames
    nzhist = np.where(hist > 0)[0]
    nzhist = nzhist[nzhist > filterlength]

    # Collapse any spatial dimensions of the stimulus array
    cstim = stimulus.reshape(-1, stimulus.shape[-1])

    # Preallocate STA array
    sta = np.zeros((cstim.shape[0], filterlength))

    # Add filterlength frames preceding each spike to the running STA
    for idx in nzhist:
        sta += hist[idx] * cstim[:, idx - filterlength : idx]

    # Mean-subtract and normalize as a vector
    sta -= np.mean(sta)
    sta /= np.linalg.norm(sta)

    # Construct a time axis to return
    tax = time[:filterlength] - time[0]
    
    # Reshape the STA and flip the time axis so that the time of the spike is at index 0
    sta = np.reshape(sta, stimulus.shape[:-1] + (filterlength,))
    sta = np.take(sta, np.arange(filterlength - 1, -1, -1), axis=-1)

    # Return STA and the time axis
    return sta, tax

def getstc(time, stimulus, spikes, filterlength):
    '''    
    Compute the spike-triggered covariance

    Usage: U, sigma, stimcov, spkcov, tax = getstc(time, stimulus, spikes, filterlength)

    Input
    -----

    time (ndarray):
        The time axis of the stimulus

    stimulus (ndarray):
        The stimulus array. The last dimension of the stimulus
        array is assumed to be time, but no other restrictions
        are placed on its shape. It works for purely temporal
        and spatiotemporal stimuli.

    spikes (ndarray):
        Array of spike times.

    filterlength (int):
        Number of frames over which to construct the
        ensemble

    Output
    ------

    note: for the following, we define the dimensionality d to be:
          d = stimulus.shape[:-1] * filterlength
          (the dimensionality of the spatiotemporal filter)

    U (ndarray):
        The (d x d) set of eigenvectors of the normalized STC matrix, each column is a separate eigenvector

    sigma (ndarray):
        The corresponding set of d eigenvalues of the normalized STC matrix

    stimcov (ndarray):
        The (d by d) stimulus covariance matrix

    spkcov (ndarray):
        The (d by d) spike-triggered covariance matrix.

    tax (ndarray):
        The time axis of the ensemble. It is of length `filterlength`,
        with intervals given by the sample rate of the `time` input
        array.

    '''

    # Bin spikes
    (hist, bins) = np.histogram(spikes, time)

    # Get indices of non-zero firing, truncating spikes earlier
    # than `filterlength` frames
    nzhist = np.where(hist > 0)[0]
    nzhist = nzhist[nzhist > filterlength]

    # Collapse any spatial dimensions of the stimulus array
    cstim = stimulus.reshape(-1, stimulus.shape[-1])

    # Preallocate STA array and STC matrix
    sta = np.zeros((cstim.shape[0], filterlength))
    spkcov = np.zeros((cstim.shape[0] * filterlength, cstim.shape[0]*filterlength))

    # Add the outerproduct of stimulus slices to the STC, keep track of the STA
    for idx in nzhist:

        # update the spike-triggered average
        sta += hist[idx] * cstim[:, idx - filterlength : idx]

        # update the spike-triggered covariance
        stimslice = (hist[idx] * cstim[:, idx - filterlength : idx]).reshape(-1,1)
        spkcov += stimslice.dot(stimslice.T)

    # Construct a time axis to return
    tax = time[:filterlength] - time[0]

    # compute the STA outer product
    sta_op = sta.reshape(-1,1).dot(sta.reshape(1,-1))

    # normalize the STC by the number of samples
    spkcov = (spkcov - sta_op) / nzhist.size

    # get the stimulus covariance matrix
    stimcov, _ = getcov(stimulus, filterlength)

    # estimate eigenvalues and eigenvectors of the normalized STC matrix
    try:
        eigvals, eigvecs = np.linalg.eig(spkcov - stimcov)
        eigvecs = np.flipud(eigvecs)
    except np.linalg.LinAlgError:
        print('Warning: eigendecomposition did not converge. You may have limited data.')
        eigvals = None
        eigvecs = None
    
    # Return values, flipped such that time of a spike is at time 0
    return eigvecs, eigvals, np.flipud(stimcov), np.flipud(spkcov), sta, tax

def lowranksta(f, k=10):
    '''
    
    Constructs a rank-k approximation to the given spatiotemporal filter.
    This is useful for computing a spatial and temporal kernel of an STA,
    or for denoising.

    Input
    -----

    f (ndarray):
        3-D filter to be separated

    k (int):
        number of components to keep (rank of the filter)

    Output
    ------

    fk (ndarray):
        the rank-k filter

    u (ndarray):
        the top k spatial components  (each row is a component)

    s (ndarray):
        the top k singular values

    u (ndarray):
        the top k temporal components (each column is a component)

    '''

    # Compute the SVD of the full filter
    try:
        u, s, v = np.linalg.svd(f.reshape(-1, f.shape[-1]), full_matrices=False)
    except LinAlgError:
        print('The SVD did not converge for the given spatiotemporal filter')
        print('The data is likely too noisy to compute a rank-{0} approximation'.format(k))
        print('Try reducing the requested rank.')
        return None, None, None, None

    # Keep the top k components
    k = np.min([k, s.size])

    # Compute the rank-k filter
    fk = (u[:,:k].dot(np.diag(s[:k]).dot(v[:k,:]))).reshape(f.shape)

    # Return the rank-k approximate filter, and the SVD components
    return fk, u, s, v

def decompose(sta):
    '''
    
    Decomposes a spatiotemporal STA into a spatial and temporal kernel

    Input
    -----

    sta (ndarray):
        The full 3-dimensional STA to be decomposed

    Output
    ------

    s (ndarray):
        The spatial kernel

    t (ndarray):
        The temporal kernel

    '''
    _, u, _, v = lowranksta(sta, k=1)
    return u[:, 0].reshape(sta.shape[:2]), v[0, :]

def _fit2Dgaussian(histogram, numSamples=1e4):
    ''' Fit 2D gaussian to empirical histogram '''

    # Indices
    x       = np.linspace(0,1,histogram.shape[0])
    y       = np.linspace(0,1,histogram.shape[1])
    xx, yy  = np.meshgrid(x, y)

    # Draw samples
    indices     = np.random.choice(np.flatnonzero(histogram+1), size=numSamples, replace=True, p=histogram.ravel())
    x_samples   = xx.ravel()[indices]
    y_samples   = yy.ravel()[indices]

    # Fit mean / covariance
    samples     = np.array((x_samples,y_samples))
    centerIdx   = np.unravel_index(np.argmax(histogram), histogram.shape)
    center      = (xx[centerIdx], yy[centerIdx])
    C           = np.cov(samples)

    # Get width / angles
    widths,vectors  = np.linalg.eig(C)
    angle           = np.arccos(vectors[0,0])

    return center, widths, angle

def _im2hist(data, spatialSmoothing = 2.5):
    ''' Converts 2D image to histogram '''

    # Smooth the data
    data_smooth = gaussian_filter(data, spatialSmoothing, order=0)

    # Mean subtract
    mu              = np.median(data_smooth)
    data_centered   = data_smooth - mu

    # Figure out if it is an on or off profile
    if np.abs(np.max(data_centered)) < np.abs(np.min(data_centered)):

        # flip from 'off' to 'on'
        data_centered *= -1;

    # Min-subtract
    data_centered -= np.min(data_centered)

    # Normalize to a PDF
    pdf = data_centered / np.sum(data_centered)

    return pdf

def getellipseparams(staframe):
    '''

    Fit an ellipse to the given spatial receptive field, return parameters of the fit ellipse

    Input
    -----

    staframe (ndarray):
        The spatial receptive field to which the ellipse should be fit

    scale (float):
        Scale factor for the ellipse

    Output
    ------

    center (tuple of floats):
        The receptive field center (location stored as an (x,y) tuple)

    widths (list of floats):
        Two-element list of the size of each principal axis of the RF ellipse

    theta (float):
        angle of rotation of the ellipse from the vertical axis, in radians

    '''

    # Get ellipse parameters
    histogram               = _im2hist(staframe)
    center, widths, theta,  = _fit2Dgaussian(histogram, numSamples=1e5)

    return center, widths, theta

def getellipse(staframe, scale=1.5):
    '''
    
    Fit an ellipse to the given spatial receptive field

    Input
    -----

    staframe:
        The spatial receptive field to which the ellipse should be fit

    scale:
        Scale factor for the ellipse

    Output
    ------

    ell:
        A matplotlib.patches.Ellipse object

    '''

    # Get ellipse parameters
    center, widths, theta = getellipseparams(staframe)

    # Generate ellipse
    ell = Ellipse(xy=center, width=scale*widths[0], height=scale*widths[1], angle=np.rad2deg(theta)+90)

    return ell

def filterpeak(sta):
    '''
    
    Find the peak (single point in space/time) of a smoothed filter

    Input
    -----

    sta (ndarray):
        Filter of which to find the peak

    Output
    ------

    idx (int):
        Linear index of the maximal point

    sidx (int), tidx (int):
        Spatial and temporal indices of the maximal point

    '''
    # Smooth filter
    fs = smoothfilter(sta, spacesig=0.7, timesig=1)

    # Find the index of the maximal point
    idx = np.unravel_index(np.abs(fs).argmax(), fs.shape)

    # Split into spatial/temporal indices
    sidx = np.roll(idx[:2], 1)
    tidx = idx[-1]

    # Return the indices
    return idx, sidx, tidx

def smoothfilter(f, spacesig=0.5, timesig=1):
    '''
    
    Smooths a 3D spatiotemporal linear filter using a multi-dimensional
    Gaussian filter with the given properties.

    Input
    -----

    f:
        3D filter to be smoothed
    
    spacesig, timesig:
        The spatial and temporal standard deviations of the Gaussian
        filter used to smooth the given filter

    Output
    ------

    fsmooth:
        The smoothed filter, with the same shape as the input
    
    '''
    return gaussian_filter(f, (spacesig, spacesig, timesig), order=0)

def cutout(arr, idx, width=5):
    '''
    
    Cut out a chunk of the given stimulus or filter

    Input
    -----

    arr (ndarray):
        Stimulus or filter array from which the chunk is cut out. The array
        should be shaped as (time, pix, pix).

    idx (array_like):
        2D array-like, specifying the row and column indices of
        the center of the section to be cut out

    width (int):
        The size of the chunk to cut out from the start indices

    Output
    ------

    cut (ndarray):
        The cut out section of the given stimulus or filter

    '''

    # Check idx is a 2-elem array-like
    if len(idx) != 2:
        raise ValueError

    # Find the indices
    row = np.arange(idx[0] - width, idx[0] + width + 1)
    col = np.arange(idx[1] - width, idx[1] + width + 1)

    # Make sure the indices are within the bounds of the given array
    row = row[(row >= 0) & (row < arr.shape[-2])]
    col = col[(col >= 0) & (col < arr.shape[-1])]

    # Mesh the indices
    rmesh, cmesh = np.meshgrid(row, col)

    # Extract and return the reduced array
    return arr[:, rmesh, cmesh]

def prinangles(u, v):
    '''

    Compute the principal angles between two subspaces. Useful for comparing 
    subspaces returned via spike-triggered covariance, for example.

    Input
    -----

    u, v (ndarray's):
        The subspaces to compare. They should be of the same size.

    Output
    ------

    ang (float):
        The angles between each dimension of the subspaces

    mag (ndarray):
        The magnitude of the overlap between each dimension of the subspace.

    '''

    # Orthogonalize each subspace
    (Qu, Ru), (Qv, Rv) = np.linalg.qr(u), np.linalg.qr(v)

    # Compute singular values of the inner product between the orthogonalized spaces
    mag = np.linalg.svd(Qu.T.dot(Qv), compute_uv=False, full_matrices=False)

    # Compute the angles between each dimension
    ang = np.rad2deg(np.arccos(mag))

    return ang, mag
