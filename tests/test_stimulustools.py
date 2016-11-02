"""test_stimulustools.py
Function definitions for testing the `pyret.stimulustools` module.
(C) 2016 The Baccus Lab
"""

import pytest
import numpy as np

from pyret import stimulustools

def test_resampling_1d():
    """Test up- and down-sampling a 1D stimulus."""
    np.random.seed(0)
    stim_size = 1000
    resample_factor = 3
    dt = 0.1
    stim = np.random.randn(stim_size,)
    time = np.arange(stim_size) * dt
    stim_us, time_us = stimulustools.upsample_stim(
            stim, resample_factor, time=time)
    stim_ds, time_ds = stimulustools.downsample_stim(
            stim_us, resample_factor, time=time_us)

    assert np.all(stim == stim_us[::resample_factor]), 'Upsampling failed'
    assert np.all(stim == stim_ds), 'Downsampling failed'

def test_resampling_2d():
    """Test up- and down-sampling a 2D stimulus."""
    np.random.seed(0)
    stim_size = (1000, 5)
    resample_factor = 3
    dt = 0.1
    stim = np.random.randn(*stim_size)
    time = np.arange(stim_size[0]) * dt
    stim_us, time_us = stimulustools.upsample_stim(
            stim, resample_factor, time=time)
    stim_ds, time_ds = stimulustools.downsample_stim(
            stim_us, resample_factor, time=time_us)

    assert np.all(stim == stim_us[::resample_factor, ...]), 'Upsampling failed'
    assert np.all(stim == stim_ds), 'Downsampling failed'

def test_slicestim_1d():
    """Test slicing a 1D stimulus into overlapping segments."""
    np.random.seed(0)
    stim_size = 1000
    stim = np.random.randn(stim_size,)
    history = 10
    sliced_stim = stimulustools.slicestim(stim, history)

    for i in range(stim_size - history):
        assert np.all(sliced_stim[i] == stim[i:i + history]), 'slicing failed'

def test_slicestim_3d():
    """Test slicing a 3D stimulus into overlapping segments."""
    np.random.seed(0)
    stim_size = (100, 5, 5)
    stim = np.random.randn(*stim_size)
    history = 10
    
    sliced_stim = stimulustools.slicestim(stim, history)
    assert sliced_stim.ndim == stim.ndim + 1
    assert sliced_stim.shape[0] == stim.shape[0] - history + 1

    for i in range(stim_size[0] - history):
        assert np.all(sliced_stim[i] == stim[i:i + history, ...]), 'slicing failed'

def test_cov():
    """Test recovering a stimulus covariance matrix."""
    np.random.seed(0)
    stim = np.random.randn(10, 2)
    assert np.allclose(np.cov(stim.T), stimulustools.cov(stim, 1))
