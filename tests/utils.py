"""utils.py
Some general utilities used in various testing routines.
(C) 2016 The Baccus Lab
"""

import os

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from pyret.filtertools import _gaussian_function
from pyret import spiketools
from pyret import visualizations

def get_image_filenames():
    """Return the filenames of all images used in the visualizations tests."""
    return (
        )

def get_image_dir():
    """Return the directory containing all baseline and test images."""
    return os.path.join(os.path.dirname(__file__), 'test-images')


def get_baseline_image_dir():
    """Return the directory containing all baseline images."""
    return os.path.join(get_image_dir(), 'baseline')


def get_test_image_dir():
    """Return the directory containing all test images."""
    return os.path.join(get_image_dir(), 'test')


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


def temporal_filter_saver(path):
    """Save the output of pyret.visualizations.temporal to the given file."""
    sta = create_default_fake_filter()[-1]
    time = np.arange(sta.shape[0])
    visualizations.temporal(time, sta)
    plt.savefig(path)
    plt.close(plt.gcf())


def spatial_filter_saver(path):
    """Save the output of pyret.visualizations.spatial to the given file."""
    sta = create_default_fake_filter()[-1]
    visualizations.spatial(sta)
    plt.savefig(path)
    plt.close(plt.gcf())


def temporal_from_spatiotemporal_filter_saver(path):
    """Save the output of pyret.visualizations.plot_sta when given only
    a temporal filter to the given file.
    """
    temporal = create_default_fake_filter()[0]
    time = np.arange(temporal.size)
    visualizations.plot_sta(time, temporal)
    plt.savefig(path)
    plt.close(plt.gcf())


def spatial_from_spatiotemporal_filter_saver(path):
    """Save the output of pyret.visualizations.plot_sta when given only
    a spatial filter to the given file.
    """
    temporal, spatial, _ = create_default_fake_filter()
    time = np.arange(temporal.size)
    visualizations.plot_sta(time, spatial)
    plt.savefig(path)
    plt.close(plt.gcf())


def spatiotemporal_filter_saver(path):
    """Save the output of pyret.visualizations.plot_sta when given a full
    spatiotemporal filter to the given file.
    """
    temporal, _, sta = create_default_fake_filter()
    time = np.arange(temporal.size)
    visualizations.plot_sta(time, sta)
    plt.savefig(path)
    plt.close(plt.gcf())


def raster_saver(path):
    """Save the output of pyret.visualizations.raster to the given file."""
    spikes, labels = create_default_fake_spikes()
    visualizations.raster(spikes, labels)
    plt.savefig(path)
    plt.close(plt.gcf())


def psth_saver(path):
    """Save the output of pyret.visualizations.psth to the given file."""
    spikes, _ = create_default_fake_spikes()
    visualizations.psth(spikes, trial_length=5.0)
    plt.savefig(path)
    plt.close(plt.gcf())


def raster_and_psth_saver(path):
    """Save the output of pyret.visualizations.raster_and_psth to the given file."""
    spikes, _ = create_default_fake_spikes()
    visualizations.raster_and_psth(spikes, trial_length=5.0)
    plt.savefig(path)
    plt.close(plt.gcf())


def sta_movie_frame_saver(path):
    """Save the a single frame from pyret.visualizations.play_sta to the
    given file.
    """
    sta = create_default_fake_filter()[-1]
    frame = get_default_movie_frame()
    animation = visualizations.play_sta(sta)
    animation._func(frame)
    plt.savefig(path)
    plt.close(plt.gcf())


def ellipse_saver(path):
    """Save the output of pyret.visualizations.ellipse to the given file."""
    sta = create_default_fake_filter()[-1]
    visualizations.ellipse(sta)
    plt.savefig(path)
    plt.close(plt.gcf())


def plot_cells_saver(path):
    """Save the output of pyret.visualizations.plot_cells to the given file."""
    ncells = 2
    stas = [create_default_fake_filter()[-1] for _ in range(ncells)]
    np.random.seed(0)
    visualizations.plot_cells(stas)
    plt.savefig(path)
    plt.close(plt.gcf())


def play_rates_saver(path):
    """Save the a single frame from pyret.visualizations.play_rates to the
    given file.
    """
    sta = create_default_fake_filter()[-1]
    rates = create_default_fake_rates()

    fig, axes = visualizations.ellipse(sta)
    patch = plt.findobj(axes, Ellipse)[0]
    animation = visualizations.play_rates(rates, patch)
    frame = get_default_movie_frame()
    animation._func(frame)
    plt.savefig(path)
    plt.close(fig)


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

