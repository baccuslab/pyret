"""test_visualizations.py
Testing code for pyret.visualizations module.
(C) 2016 Benjamin Naecker, Niru Maheswaranathan
"""

import os

from matplotlib.testing.compare import compare_images
import numpy as np
import pytest

from pyret import visualizations as viz
from pyret import spiketools
import utils # Pyret testing and image generation utilities

def _comparison_wrapper(filename, save_func):
    """A wrapper function for comparing a test image with the baseline version.

    The actual methods which generate and save the testing (and baseline)
    images are defined in the `utils` module. These are the functions such
    as `plot_cells_saver`, which will save the output of 
    `pyret.visualizations.plot_cells` into the file with the given name.

    This method calls the saver function with the given filename, then
    compares the baseline version (which has the same name but is in the
    `test-images/baseline` directory) against the generated test image.

    Parameters
    ----------
    
    filename : str
        The name of the file (without a path) in which to save the test
        image.

    save_func : callable
        The saver function which will generate and save the test image
        in the passed `filename`.
    """
    test_name = os.path.join(utils.get_test_image_dir(), filename)
    base_name = os.path.join(utils.get_baseline_image_dir(), filename)
    save_func(test_name)
    assert not compare_images(test_name, base_name, 1)
    os.remove(test_name)


def test_temporal_filter():
    """Test plotting a temporal filter from a full 3D spatiotemporal STA."""
    _comparison_wrapper('temporal-filter.png', utils.temporal_filter_saver)


def test_spatial_filter():
    """Test plotting a spatial filter from a full 3D spatiotemporal STA."""
    _comparison_wrapper('spatial-filter.png', utils.spatial_filter_saver)


def test_spatiotemporal_filter():
    """Test plotting a full 3D spatiotemporal STA."""
    # Test plotting temporal component
    _comparison_wrapper('temporal-from-full-filter.png', 
            utils.temporal_from_spatiotemporal_filter_saver)

    # Test plotting spatial component
    _comparison_wrapper('spatial-from-full-filter.png', 
        utils.spatial_from_spatiotemporal_filter_saver)

    # Test plotting both spatial/temporal components
    _comparison_wrapper('full-filter.png',
        utils.spatiotemporal_filter_saver)


def test_raster():
    """Test plotting a spike raster."""
    _comparison_wrapper('raster.png', utils.raster_saver)


def test_psth():
    """Test plotting a PSTH."""
    _comparison_wrapper('psth.png', utils.psth_saver)


def test_raster_and_psth():
    """Test plotting a raster and PSTH on the same axes."""
    _comparison_wrapper('raster-and-psth.png', utils.raster_and_psth_saver)


def test_play_sta():
    """Test playing an STA as a movie.
    
    Matplotlib doesn't yet have a way to compare movies, and the formats
    and precise bytes written by different encoding libraries are too variable
    to test reliably. Instead, we write a specific frame of the animation
    to disk as an image, and compare it with a baseline.
    """
    _comparison_wrapper('sta-movie-frame.png', utils.sta_movie_frame_saver)


def test_ellipse():
    """Test plotting an ellipse fitted to an RF."""
    _comparison_wrapper('ellipse.png', utils.ellipse_saver)
    

def test_plot_cells():
    """Test plotting ellipses for multiple cells on the same axes."""
    _comparison_wrapper('plotcells.png', utils.plot_cells_saver)


def test_play_rates():
    """Test playing firing rates for cells as a movie."""
    _comparison_wrapper('rates-movie-frame.png', utils.play_rates_saver)


def test_anim_to_html():
    """Test converting an animation to HTML."""
    try:
        from IPython.display import HTML
    except ImportError:
        pytest.skip('Cannot convert movie to HTML without IPython.')

    sta = utils.create_default_fake_filter()[-1]
    html = viz.anim_to_html(viz.play_sta(sta))
    assert isinstance(html, HTML)

