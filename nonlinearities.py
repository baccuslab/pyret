'''
nonlinearities.py

Tools for fitting nonlinear functions to data

(c) 2014 bnaecker, nirum
'''

import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, mu, sigma):
    '''
    A 1D (unnormalized) gaussian function
    '''

    return np.exp( -0.5 * ((x-mu) / sigma)**2 )

def sigmoid(x, threshold, slope, peak, offset):
    '''
    A sigmoidal nonlinearity
    '''

    return offset + peak / (1 + np.exp(-slope*(x - threshold)))

def dprime(p0, p1):
    '''
    compute d' between two distributions given mean / standard deviation

    input
    -----
    p0: (mean, standard deviation) for the first distribution
    p1: (mean, standard deviation) for the second distribution

    '''
    return (p1[0] - p0[0]) / np.sqrt( p1[1]**2 + p0[1]**2 )

def fitgaussian(xpts, ypts, p0=None):
    '''
    Fit a gaussian function to noisy data

    input
    -----

    xpts [array]:
        x-values of the data to fit

    ypts [array]: 
        y-values of the data to fit

    output
    ------

    popt [array]:
        The best-fit sigmoidal parameters (threshold, slope, peak, and offset)

    yhat [array]:
        The estimated y-values at the given locations in xpts

    pcov [matrix]:

    '''

    # estimate initial conditions
    if p0 is None:
        p0 = (np.mean(xpts), 5*np.mean(np.diff(xpts)))

    # normalize the max to have value 1
    scalefactor = float( np.max(ypts) )
    ypts = ypts / scalefactor

    # get parameters
    popt, pcov = curve_fit(gaussian, xpts, ypts, p0)

    # evaluate fit
    yhat = gaussian(xpts, *popt) * scalefactor

    return popt, yhat, pcov

def fitsigmoid(xpts, ypts):
    '''
    Fit a sigmoidal function to noisy data

    input
    -----

    xpts [array]:
        x-values of the data to fit

    ypts [array]: 
        y-values of the data to fit

    output
    ------

    popt [array]:
        The best-fit sigmoidal parameters (threshold, slope, peak, and offset)

    yhat [array]:
        The estimated y-values at the given locations in xpts

    pcov [matrix]:

    '''

    # estimate initial conditions
    p0 = (np.mean(xpts), 1, np.max(ypts), np.min(ypts))

    # get parameters
    popt, pcov = curve_fit(sigmoid, xpts, ypts, p0)

    # evaluate fit
    yhat = sigmoid(xpts, *popt)

    return popt, yhat, pcov

def estdprime(u, r, numbins=100):
    '''
    Fit a nonlinearity given a 1D stimulus projection u and spiking response r
    '''

    # pick a set of bins, store centered bins
    bins = np.linspace(np.min(u), np.max(u), numbins)
    bc   = bins[:-1] + np.mean(np.diff(bins))*0.5

    # bin the raw stimulus distribution
    raw, _ = np.histogram(u, bins)

    # bin the spike-triggered distribution
    data = u[np.where(r > 0)[0]]
    spk, _ = np.histogram(data, bins)

    # estimate gaussian parameters
    try:
        raw_params = fitgaussian(bc, raw, (np.mean(u), np.std(u)))[0]
        spk_params = fitgaussian(bc, spk, (np.mean(data), np.std(data)))[0]
    except RuntimeError:
        print('Warning: Gaussian curve fit did not converge')
        raw_params = (np.mean(u), np.std(u))
        spk_params = (np.mean(data), np.std(data))

    # estimate d'
    return dprime(raw_params, spk_params)

def estnln(u, r, numbins=50):
    '''
    Fit a nonlinearity given a 1D stimulus projection u and spiking response r
    '''

    # the minimum number of data points / bin to keep for fitting
    mincount = 2

    # bin the raw stimulus distribution
    raw, bins = np.histogram(u, numbins)

    # bin the spike-triggered distribution
    spk, _    = np.histogram(u[np.where(r > 0)[0]], bins)

    # find locations where there are enough data points
    locs = np.logical_and( (raw > mincount), (spk > mincount) )

    # normalize the two distributions
    raw = raw / float(np.sum(raw))
    spk = spk / float(np.sum(spk))

    # take the ratio of the two distributions
    ratio = spk[locs] / raw[locs]
    xvals = bins[locs]

    # fit a sigmoid to the results
    popt, yhat, pcov = fitsigmoid(xvals, ratio)

    # return values
    return popt, xvals, ratio, yhat, pcov
