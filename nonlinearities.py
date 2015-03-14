"""
Tools for fitting nonlinear functions to data

.. warning:: These functions have not been tested

"""

import numpy as _np
from scipy.optimize import curve_fit


def gaussian(x, mu, sigma):
    """
    A 1D (unnormalized) gaussian function

    """

    return _np.exp( -0.5 * ((x-mu) / sigma)**2 )


def sigmoid(x, threshold, slope, peak, offset):
    """
    A sigmoidal nonlinearity

    """

    return offset + peak / (1 + _np.exp(-slope*(x - threshold)))


def dprime(p0, p1):
    """
    compute d' between two distributions given mean / standard deviation

    Parameters
    ----------

    p0 : (float, float)
        Mean and standard deviation for the first distribution

    p1 : (float, float)
        Mean and standard deviation for the second distribution

    """
    return (p1[0] - p0[0]) / _np.sqrt( p1[1]**2 + p0[1]**2 )


def fitgaussian(xpts, ypts, p0=None):
    """
    Fit a gaussian function to noisy data

    Parameters
    ----------

    xpts : array_like
        x-values of the data to fit

    ypts : array_like
        y-values of the data to fit

    Returns
    -------

    popt : array_like
        The best-fit sigmoidal parameters (threshold, slope, peak, and offset)

    yhat : array_like
        The estimated y-values at the given locations in xpts

    pcov [matrix]:

    """

    # estimate initial conditions
    if p0 is None:
        p0 = (_np.mean(xpts), 5*_np.mean(_np.diff(xpts)))

    # normalize the max to have value 1
    scalefactor = float( _np.max(ypts) )
    ypts = ypts / scalefactor

    # get parameters
    popt, pcov = curve_fit(gaussian, xpts, ypts, p0)

    # evaluate fit
    yhat = gaussian(xpts, *popt) * scalefactor

    return popt, yhat, pcov


def fitsigmoid(xpts, ypts):
    """
    Fit a sigmoidal function to noisy data

    Parameters
    ----------

    xpts : array_like
        x-values of the data to fit

    ypts : array_like
        y-values of the data to fit

    Returns
    -------

    popt : array_like
        The best-fit sigmoidal parameters (threshold, slope, peak, and offset)

    yhat : array_like
        The estimated y-values at the given locations in xpts

    pcov : array_like

    """

    # estimate initial conditions
    p0 = (_np.mean(xpts), 1, _np.max(ypts), _np.min(ypts))

    # get parameters
    popt, pcov = curve_fit(sigmoid, xpts, ypts, p0)

    # evaluate fit
    yhat = sigmoid(xpts, *popt)

    return popt, yhat, pcov


def estdprime(u, r, numbins=100):
    """
    Fit a nonlinearity given a 1D stimulus projection u and spiking response r

    """

    # pick a set of bins, store centered bins
    bins = _np.linspace(_np.min(u), _np.max(u), numbins)
    bc   = bins[:-1] + _np.mean(_np.diff(bins))*0.5

    # bin the raw stimulus distribution
    raw, _ = _np.histogram(u, bins)

    # bin the spike-triggered distribution
    data = u[r > 0]
    spk, _ = _np.histogram(data, bins)

    # estimate gaussian parameters
    try:
        raw_params = fitgaussian(bc, raw, (_np.mean(u), _np.std(u)))[0]
        spk_params = fitgaussian(bc, spk, (_np.mean(data), _np.std(data)))[0]
    except RuntimeError:
        print('Warning: Gaussian curve fit did not converge')
        raw_params = (_np.mean(u), _np.std(u))
        spk_params = (_np.mean(data), _np.std(data))

    # estimate d'
    return dprime(raw_params, spk_params)


def estnln(u, r, numbins=50):
    """
    Fit a nonlinearity given a 1D stimulus projection u and spiking response r

    """

    # the minimum number of data points / bin to keep for fitting
    mincount = 2

    # bin the raw stimulus distribution
    raw, bins = _np.histogram(u, numbins)

    # bin the spike-triggered distribution
    spk, _    = _np.histogram(u[r > 0], bins)

    # find locations where there are enough data points
    locs = _np.logical_and( (raw > mincount), (spk > mincount) )

    # normalize the two distributions
    raw = raw / float(_np.sum(raw))
    spk = spk / float(_np.sum(spk))

    # take the ratio of the two distributions
    ratio = spk[locs] / raw[locs]
    xvals = bins[locs]

    # fit a sigmoid to the results
    popt, yhat, pcov = fitsigmoid(xvals, ratio)

    # return values
    return popt, xvals, ratio, yhat, pcov
