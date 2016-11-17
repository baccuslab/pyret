"""test_visualizations.py
Testing code for pyret.visualizations module.
(C) 2016 Benjamin Naecker, Niru Maheswaranathan
"""

import os

from matplotlib.testing.compare import compare_images
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyret import visualizations as viz
from pyret import spiketools
import utils # Pyret testing utilities

IMG_DIR = os.path.join(os.path.dirname(__file__), 'test-images')

def test_temporal_filter():
    """Test plotting a temporal filter from a full 3D spatiotemporal STA."""
    nx, ny, nt = 10, 10, 50
    time = np.arange(nt)
    sta = utils.create_spatiotemporal_filter(nx, ny, nt)[-1]

    viz.temporal(time, sta)
    filename = os.path.join(IMG_DIR, 'test-temporal-filter.png')
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-temporal-filter.png'), filename, 1)
    os.remove(filename)
    plt.close('all')


def test_spatial_filter():
    """Test plotting a spatial filter from a full 3D spatiotemporal STA."""
    nx, ny, nt = 10, 10, 50
    time = np.arange(nt)
    sta = utils.create_spatiotemporal_filter(nx, ny, nt)[-1]
    viz.spatial(sta)
    filename = os.path.join(IMG_DIR, 'test-spatial-filter.png')
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-spatial-filter.png'), filename, 1)
    os.remove(filename)
    plt.close('all')


def test_spatiotemporal_filter():
    """Test plotting a full 3D spatiotemporal STA."""
    nx, ny, nt = 10, 10, 50
    time = np.arange(nt)
    t, s, sta = utils.create_spatiotemporal_filter(nx, ny, nt)

    # Test plotting temporal component
    filename = os.path.join(IMG_DIR, 'test-temporal-filter.png')
    viz.plot_sta(time, t)
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-temporal-from-spatiotemporal-filter.png'), 
            filename, 1)
    os.remove(filename)
    plt.close('all')

    # Test plotting spatial component
    filename = os.path.join(IMG_DIR, 'test-temporal-filter.png')
    viz.plot_sta(time, s)
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-spatial-from-spatiotemporal-filter.png'), 
            filename, 1)
    os.remove(filename)
    plt.close('all')

    # Test plotting both spatial/temporal components
    filename = os.path.join(IMG_DIR, 'test-full-spatiotemporal-filter.png')
    viz.plot_sta(time, sta)
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-full-spatiotemporal-filter.png'), filename, 1)
    os.remove(filename)
    plt.close('all')


def test_raster():
    """Test plotting a spike raster."""
    spikes = np.arange(10)
    labels = np.array((1, 1, 1, 1, 1, 2, 2, 2, 2, 2))
    fig, axes = viz.raster(spikes, labels)
    filename = os.path.join(IMG_DIR, 'test-raster.png')
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-raster.png'), filename, 1)
    os.remove(filename)
    plt.close('all')


def test_psth():
    """Test plotting a PSTH."""
    spikes = np.arange(10)
    fig, axes = viz.psth(spikes, trial_length=5.)
    filename = os.path.join(IMG_DIR, 'test-psth.png')
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-psth.png'), filename, 1)
    os.remove(filename)
    plt.close('all')


def test_raster_and_psth():
    """Test plotting a raster and PSTH on the same axes."""
    spikes = np.arange(10)
    fig, axes = viz.raster_and_psth(spikes, trial_length=5.)
    filename = os.path.join(IMG_DIR, 'test-raster-and-psth.png')
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-raster-and-psth.png'), filename, 1)
    os.remove(filename)
    plt.close('all')


def test_play_sta():
    """Test playing an STA as a movie.
    
    Matplotlib doesn't yet have a way to compare movies, and the formats
    and precise bytes written by different encoding libraries are too variable
    to test reliably. Instead, we write a specific frame of the animation
    to disk as an image, and compare it with a baseline.
    """
    nx, ny, nt = 10, 10, 50
    sta = utils.create_spatiotemporal_filter(nx, ny, nt)[-1]
    anim = viz.play_sta(sta)
    filename = os.path.join(IMG_DIR, 'test-sta-movie.png')
    frame = 10
    anim._func(frame)
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-sta-movie-frame.png'),
            filename, 1)
    os.remove(filename)
    plt.close('all')

def test_ellipse():
    """Test plotting an ellipse fitted to an RF."""
    nx, ny, nt = 10, 10, 50
    sta = utils.create_spatiotemporal_filter(nx, ny, nt)[-1]
    filename = os.path.join(IMG_DIR, 'test-ellipse.png')
    viz.ellipse(sta)
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-ellipse.png'), filename, 1)
    os.remove(filename)
    plt.close('all')


def test_plot_cells():
    """Test plotting ellipses for multiple cells on the same axes."""
    nx, ny, nt = 10, 10, 50
    stas = []
    ncells = 2
    for cell in range(ncells):
        stas.append(utils.create_spatiotemporal_filter(nx, ny, nt)[-1])

    filename = os.path.join(IMG_DIR, 'test-plotcells.png')
    np.random.seed(0) # plot_cells() uses random colors for each cell
    viz.plot_cells(stas)
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-plotcells.png'), filename, 1)
    os.remove(filename)
    plt.close('all')


def test_play_rates():
    """Test playing firing rates for cells as a movie."""
    nx, ny, nt = 10, 10, 50
    sta = utils.create_spatiotemporal_filter(nx, ny, nt)[-1]
    time = np.linspace(0, 10, 100)
    spikes = np.arange(10)
    binned_spikes = spiketools.binspikes(spikes, time)
    rate = spiketools.estfr(binned_spikes, time)

    # Plot cell
    fig, axes = viz.ellipse(sta)
    patch = plt.findobj(axes, Ellipse)[0]
    anim = viz.play_rates(rate, patch)
    filename = os.path.join(IMG_DIR, 'test-rates-movie.png')
    frame = 10
    anim._func(frame)
    plt.savefig(filename)
    assert not compare_images(
            os.path.join(IMG_DIR, 'baseline-rates-movie-frame.png'),
            filename, 1)
    os.remove(filename)
    plt.close('all')


def test_to_html():
    """Test converting an animation to HTML."""
    try:
        from IPython.display import HTML
    except ImportError:
        pytest.skip('Cannot convert movie to HTML without IPython.')

    nx, ny, nt = 10, 10, 50
    sta = utils.create_spatiotemporal_filter(nx, ny, nt)[-1]
    anim = viz.play_sta(sta)
    html = viz.to_html(anim)
    assert isinstance(html, HTML)

