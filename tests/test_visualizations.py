"""test_visualizations.py
Testing code for pyret.visualizations module.
(C) 2016 Benjamin Naecker, Niru Maheswaranathan
"""

import os

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.patches import Ellipse
import numpy as np
import pytest

from pyret import visualizations
from pyret import spiketools
import utils # Pyret testing utilities


def test_temporal_filter():
    """Test plotting a temporal filter directly."""
    # Plot filter
    temporal_filter, _, _ = utils.create_default_fake_filter()
    time = np.arange(temporal_filter.size)
    visualizations.temporal(time, temporal_filter)

    # Verify data plotted correctly
    line = plt.findobj(plt.gca(), plt.Line2D)[0]
    assert np.all(line.get_xdata() == time), \
            'Time axis data is incorrect.'
    assert np.all(line.get_ydata() == temporal_filter), \
            'Temporal filter data is incorrect.'
    plt.close(plt.gcf())


def test_spatial_filter():
    """Test plotting a spatial filter directly."""
    # Plot filter
    _, spatial_filter, _ = utils.create_default_fake_filter()
    visualizations.spatial(spatial_filter)
    data = spatial_filter - spatial_filter.mean()

    # Verify data plotted correctly
    img = plt.findobj(plt.gca(), AxesImage)[0]
    assert np.all(img.get_array() == data), 'Spatial filter data is incorrect.'
    plt.close(plt.gcf())

    # Verify data plotted correctly when giving a maximum value
    maxval = np.abs(spatial_filter).max()
    visualizations.spatial(spatial_filter, maxval=maxval)
    img = plt.findobj(plt.gca(), AxesImage)[0]
    assert np.all(img.get_array() == spatial_filter), \
            'Spatial filter data incorrect when passing explicit maxval'
    assert np.all(img.get_clim() == np.array((-maxval, maxval))), \
            'Spatial filter color limits not set correctly.'
    plt.close(plt.gcf())


def test_plot_sta():
    """Test visualizations.plot_sta method."""
    # Test plotting temporal component
    temporal_filter, spatial_filter, sta = utils.create_default_fake_filter()
    time = np.arange(temporal_filter.size)
    visualizations.plot_sta(time, temporal_filter)
    line = plt.findobj(plt.gca(), plt.Line2D)[0]
    assert np.all(line.get_xdata() == time), 'Time axis data is incorrect.'
    assert np.all(line.get_ydata() == temporal_filter), 'Temporal filter data is incorrect.'
    plt.close(plt.gcf())

    # Test plotting spatial component
    visualizations.plot_sta(time, spatial_filter)
    img = plt.findobj(plt.gca(), AxesImage)[0]
    desired = (spatial_filter - spatial_filter.mean()) / spatial_filter.var()
    actual = img.get_array()
    assert np.allclose(actual, desired), 'Spatial filter data is incorrect.'
    plt.close(plt.gcf())

    # Test plotting both spatial/temporal components.
    # This code is a bit suspect. `plot_sta` internally calls 
    # `filtertools.decompose`, which will find singular vectors that are
    # unit norm. But then `plot_sta` also calls `spatial`, which does
    # some of its own normalization. The result is that it's difficult
    # to know what scale the true data plotted should have, so this test
    # just normalizes all plots and images.
    fig, axes = visualizations.plot_sta(time, sta)
    img = plt.findobj(axes[0], AxesImage)[0]
    desired = (spatial_filter - spatial_filter.mean())
    desired /= desired.max()
    actual = img.get_array()
    actual /= actual.max()
    assert np.allclose(actual, desired), 'Spatial filter data is incorrect.'

    line = plt.findobj(axes[1], plt.Line2D)[0]
    assert np.all(line.get_xdata() == time), 'Time axis data is incorrect.'
    desired = (temporal_filter - temporal_filter.min())
    desired /= desired.max()
    actual = line.get_ydata()
    actual -= actual.min()
    actual /= actual.max()
    assert np.allclose(desired, actual), 'Temporal filter data is incorrect.'

    # Verify raising a value error when incorrect dimensionality passed
    with pytest.raises(ValueError):
        visualizations.plot_sta(None, np.random.randn(2, 2, 2, 2))


def test_raster():
    """Test plotting a spike raster."""
    spikes, labels = utils.create_default_fake_spikes()
    visualizations.raster(spikes, labels)
    line = plt.findobj(plt.gca(), plt.Line2D)[0]
    assert np.all(line.get_xdata() == spikes), 'Spike times do not match'
    assert np.all(line.get_ydata() == labels), 'Spike labels do not match'

    # Verify exception raised when spikes and labels different length
    with pytest.raises(AssertionError):
        visualizations.raster(np.array((0, 1)), np.array((0, 1, 2)))


def test_psth():
    """Test plotting a PSTH."""
    spikes, _ = utils.create_default_fake_spikes()
    visualizations.psth(spikes, trial_length=5.0)
    line = plt.findobj(plt.gca(), plt.Line2D)[0]
    xdata, ydata = line.get_data()
    binsize = 0.01
    ntrials = 2
    assert xdata.size == (spikes.size / binsize) / ntrials, \
            'visualizations.psth did not use trial length correctly'
    assert ydata.max() == (1 / binsize), \
            'visualizations.psth did not correctly compute max spike rate'


def test_raster_and_psth():
    """Test plotting a raster and PSTH on the same axes."""
    spikes, _ = utils.create_default_fake_spikes()
    visualizations.raster_and_psth(spikes, trial_length=5.0)
    axes = plt.findobj(plt.gcf(), plt.Axes)
    psth_line = plt.findobj(axes[0], plt.Line2D)[0]
    raster_lines = plt.findobj(axes[1], plt.Line2D)[:2]
    
    binsize = 0.01
    ntrials = 2
    assert psth_line.get_xdata().size == (spikes.size / binsize) / ntrials, \
            'visualizations.raster_and_psth did not use trial length correctly'
    assert psth_line.get_ydata().max() == (1 / binsize), \
            'visualizations.raster_and_psth did not correctly compute max spike rate'
    assert raster_lines[0].get_xdata().size == 5, \
            'visualizations.raster_and_psth did not correctly split rasters into trials'
    assert raster_lines[0].get_ydata().size == 5, \
            'visualizations.raster_and_psth did not correctly split rasters into trials'
    assert np.all(raster_lines[0].get_ydata() == 0.0), \
            'visualizations.raster_and_psth did not correctly label rasters'
    assert raster_lines[1].get_xdata().size == 4, \
            'visualizations.raster_and_psth did not correctly split rasters into trials'
    assert raster_lines[1].get_ydata().size == 4, \
            'visualizations.raster_and_psth did not correctly split rasters into trials'
    assert np.all(raster_lines[1].get_ydata() == 1.0), \
            'visualizations.raster_and_psth did not correctly label rasters'

def test_play_sta():
    """Test playing an STA as a movie by comparing a known frame."""
    sta = utils.create_default_fake_filter()[-1]
    sta -= sta.mean()
    frame = utils.get_default_movie_frame()
    animation = visualizations.play_sta(sta)
    animation._func(frame)
    imgdata = plt.findobj(plt.gcf(), AxesImage)[0].get_array()
    imgdata -= imgdata.mean()
    data = sta[frame, ...]
    data -= data.mean()
    assert np.allclose(imgdata, data), \
            'visualizations.play_sta did not animate the 3D sta correctly.'


def test_ellipse():
    """Test plotting an ellipse fitted to an RF."""
    temporal_filter, spatial_filter, sta = utils.create_default_fake_filter()
    fig, ax = visualizations.ellipse(sta)
    el = plt.findobj(ax, Ellipse)[0]
    assert np.allclose(el.center, np.array(spatial_filter.shape) / 2.0), \
            'visualizations.ellipse did not compute correct ellipse center'
    assert np.allclose((el.height, el.width), 2.827082246), \
            'visualizations.ellipse computed incorrect width and/or height'
    

def test_plot_cells():
    """Test plotting ellipses for multiple cells on the same axes."""
    ncells = 2
    stas = [utils.create_default_fake_filter()[-1] for 
            _ in range(ncells)]
    np.random.seed(0)
    visualizations.plot_cells(stas)

    ellipses = plt.findobj(plt.gca(), Ellipse)
    for el in ellipses:
        assert np.allclose(el.center, utils.get_default_filter_size()[0] / 2.), \
                'visualizations.plot_cells did not compute correct ellipse center'
        assert np.allclose((el.height, el.width), 2.827082246), \
                'visualizations.plot_cells computed incorrect width and/or height'


def test_play_rates():
    """Test playing firing rates for cells as a movie."""
    sta = utils.create_default_fake_filter()[-1]
    rates = utils.create_default_fake_rates()
    fig, axes = visualizations.ellipse(sta)
    patch = plt.findobj(axes, Ellipse)[0]
    animation = visualizations.play_rates(rates, patch)

    frame = utils.get_default_movie_frame()
    animation._func(frame)
    cmap = plt.cm.gray(np.arange(255))
    desired_color = cmap[int(rates[frame] / rates.max())]
    assert np.all(patch.get_facecolor()[:3] == desired_color[:3]), \
            'visualizations.play_rates did not set patch color correctly'


def test_anim_to_html():
    """Test converting an animation to HTML."""
    try:
        from IPython.display import HTML
    except ImportError:
        pytest.skip('Cannot convert movie to HTML without IPython.')

    sta = utils.create_default_fake_filter()[-1]
    html = visualizations.anim_to_html(visualizations.play_sta(sta))
    assert isinstance(html, HTML)

