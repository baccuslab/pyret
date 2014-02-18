"""
Routines for fitting gaussian profiles to data
author: Niru Maheswaranathan
02:57 PM Feb 8, 2014
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from math import atan2
from matplotlib.patches import Ellipse

def im2hist(data, spatialSmoothing = 2.5):
    """
    Converts 2D image to histogram
    """

    # smooth the data
    data_smooth = gaussian_filter(data, spatialSmoothing, order=0)

    # mean subtract
    mu = np.median(data_smooth)
    data_centered = data_smooth - mu

    # figure out if it is an on or off profile
    if np.abs(np.max(data_centered)) < np.abs(np.min(data_centered)):

        # flip from 'off' to 'on'
        data_centered *= -1;

    # min-subtract
    data_centered -= np.min(data_centered)

    # normalize to a PDF
    pdf = data_centered / np.sum(data_centered)

    return pdf

def fit2Dgaussian(histogram, numSamples=1e4):
    """
    Fit 2D gaussian to empirical histogram
    """

    # indices
    x = np.linspace(0,1,histogram.shape[0])
    y = np.linspace(0,1,histogram.shape[1])
    xx,yy = np.meshgrid(x,y)

    # draw samples
    indices = np.random.choice(np.flatnonzero(histogram+1), size=numSamples, replace=True, p=histogram.ravel())
    x_samples = xx.ravel()[indices]
    y_samples = yy.ravel()[indices]

    # fit mean / covariance
    samples = np.array((x_samples,y_samples))
    #center = np.mean(samples, axis=1)
    centerIdx = np.unravel_index(np.argmax(histogram), histogram.shape)
    center = (xx[centerIdx], yy[centerIdx])
    C = np.cov(samples)

    # get width / angles
    widths,vectors = np.linalg.eig(C)
    angle = np.arccos(vectors[0,0])

    return center, widths, angle, xx, yy

def getEllipse(F, scale=1.5):
    """
    Return a 1D ellipse fit to an empirical pdf
    """

    # get ellipse parameters
    histogram = im2hist(F)
    center, widths, theta, xx, yy = fit2Dgaussian(histogram, numSamples=1e5)

    # generate ellipse
    ell = Ellipse(xy=center, width=scale*widths[0], height=scale*widths[1], angle=np.rad2deg(theta)+90)

    return ell

def plotEllipse(F, scale=1.5):

    # get ellipse parameters
    histogram = im2hist(F)
    center, widths, theta, xx, yy = fit2Dgaussian(histogram, numSamples=1e5)

    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    # plot the raw data
    img = ax.pcolor(xx,yy,F)
    img.set_cmap('RdBu')

    # generate ellipse
    ell = Ellipse(xy=center, width=scale*widths[0], height=scale*widths[1], angle=np.rad2deg(theta)+90)

    # add it to the plot
    ell.set_facecolor('green')
    ell.set_alpha(0.5)
    ell.set_edgecolor('black')
    ax.add_artist(ell)

    plt.show()
    plt.draw()
    return ax

def test():
    x = [5,7,11,15,16,17,18]
    y = [8, 5, 8, 9, 17, 18, 25]
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt
    ax = plt.subplot(111, aspect='equal')
    for j in xrange(1, 4):
        ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                      angle=np.rad2deg(np.arccos(v[0, 0])))
        ell.set_facecolor('cyan')
        ell.set_alpha(0.1)
        ell.set_edgecolor('black')
        ax.add_artist(ell)
    plt.scatter(x, y)
    plt.show()
