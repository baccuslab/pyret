"""utils.py
Some general utilities used in various testing routines.
(C) 2016 The Baccus Lab
"""

import os

import numpy as np

from pyret.filtertools import _gaussian_function
from pyret import spiketools


def get_default_filter_size():
    """Return the default x, y, and temporal size of a spatiotemporal filter."""
    return 10, 10, 50


def get_default_movie_frame():
    """Return the default movie frame used in testing movie-generating code."""
    return 10


def create_default_fake_filter():
    """Return the default temporal, spatial, and spatiotemporal filters."""
    nx, ny, nt = get_default_filter_size()
    time = np.arange(nt)
    return create_spatiotemporal_filter(nx, ny, nt)


def create_default_fake_spikes():
    """Return the default spike times and labels."""
    spikes = np.arange(10)
    labels = np.array((1, 1, 1, 1, 1, 2, 2, 2, 2, 2))
    return spikes, labels


def create_default_fake_rates():
    """Return the default firing rates."""
    spikes = np.arange(10)
    time = np.linspace(0, 10, 100)
    binned = spiketools.binspikes(spikes, time)
    rate = spiketools.estfr(binned, time)
    return rate


def create_temporal_filter(n, norm=True):
    """Returns a fake temporal linear filter that superficially resembles
    those seen in retinal ganglion cells. 

    Parameters
    ----------

    n : int
        Number of time points in the filter.

    norm : bool, optional
        If True, normalize the filter to have unit 2-norm. Defaults to True.

    Returns
    -------

    f : ndarray
        The fake linear filter
    """
    time_axis = np.linspace(0, 2 * np.pi, n)
    filt = np.exp(-1. * time_axis) * np.sin(time_axis)
    return filt / np.linalg.norm(filt) if norm else filt


def create_spatiotemporal_filter(nx, ny, nt, norm=True):
    """Returns a fake 3D spatiotemporal filter.

    The filter is created as the outer product of a 2D gaussian with a fake
    temporal filter as returned by `create_temporal_filter()`.

    Parameters
    ----------

    nx, ny : int
        Number of points in the two spatial dimensions of the stimulus.

    nt : int
        Number of time points in the stimulus.

    norm : bool, optional
        If True, normalize the filter to have unit 2-norm. Defaults to True.

    Returns
    -------

    t : ndarray
        The temporal filter used.

    s : ndarray
        The spatial filter used.

    f : ndarray
        The full spatiotemporal linear filter, shaped (nt, nx, ny).
    """
    temporal_filter = create_temporal_filter(nt, norm)

    grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    points = np.array([each.flatten() for each in grid])
    gaussian = _gaussian_function(points, int(ny / 2), int(nx / 2), 1, 0, 1).reshape(nx, ny)
    if norm:
        gaussian /= np.linalg.norm(gaussian)

    # Outer product
    filt = np.einsum('i,jk->ijk', temporal_filter, gaussian)

    return (temporal_filter, gaussian,
            filt / np.linalg.norm(filt) if norm else filt)

