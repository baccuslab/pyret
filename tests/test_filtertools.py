"""test_filtertools.py
Test code for pyret's filtertools module.
(C) 2016 The Baccus Lab.
"""

import numpy as np
import pytest

from pyret import filtertools as flt
from pyret.stimulustools import slicestim

import utils

def test_linear_prediction_1d():
    """Test method for computing linear prediction from a 
    filter to a one-dimensional stimulus.
    """
    filt = np.random.randn(100,)
    stim = np.random.randn(1000,)
    pred = flt.linear_prediction(filt, stim)

    sl = slicestim(stim, filt.shape[0])
    assert np.allclose(filt.reshape(1, -1).dot(sl), pred)

def test_linear_prediction_nd():
    """Test method for computing linear prediction from a 
    filter to a multi-dimensional stimulus.
    """
    for ndim in range(2, 4):
        filt = np.random.randn(100, *((10,) * ndim))
        stim = np.random.randn(1000, *((10,) * ndim))
        pred = flt.linear_prediction(filt, stim)

        sl = slicestim(stim, filt.shape[0])
        tmp = np.zeros(sl.shape[1])
        filt_reshape = filt.reshape(1, -1)
        for i in range(tmp.size):
            tmp[i] = filt_reshape.dot(sl[:, i, :].reshape(-1, 1))

        assert np.allclose(tmp, pred)


def test_linear_prediction_raises():
    """Test raising ValueErrors with incorrect inputs"""
    with pytest.raises(ValueError):
        flt.linear_prediction(np.random.randn(10,), np.random.randn(10,2))
        flt.linear_prediction(np.random.randn(10, 2), np.random.randn(10, 3))

def test_revco_1d():
    """Test computation of a 1D linear filter by reverse correlation"""
    # Create fake filter, 100 time points
    filter_length = 100
    true = utils.create_temporal_filter(filter_length)

    # Compute linear response
    stim_length = 10000
    stimulus = np.random.randn(stim_length,)
    response = flt.linear_prediction(true, stimulus)

    # Reverse correlation
    filt = flt.revco(response, stimulus, filter_length)
    tol = 0.1
    assert np.allclose(true, filt, atol=tol)


def test_revco_nd():
    """Test computation of 3D linear filter by reverse correlation"""
    # Create fake filter
    filter_length = 100
    nx, ny = 10, 10
    true = utils.create_spatiotemporal_filter(nx, ny, filter_length)

    # Compute linear response
    stim_length = 10000
    stimulus = np.random.randn(stim_length, nx, ny)
    response = flt.linear_prediction(true, stimulus)

    # Reverse correlation
    filt = flt.revco(response, stimulus, filter_length)
    tol = 0.1
    assert np.allclose(true, filt, atol=tol)
