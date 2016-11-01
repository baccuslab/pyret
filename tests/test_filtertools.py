"""test_filtertools.py
Test code for pyret's filtertools module.
(C) 2016 The Baccus Lab.
"""

import numpy as np
import pytest

from pyret import filtertools as flt
from pyret.stimulustools import slicestim


def test_linear_prediction_one_dim():
    """Test method for computing linear prediction from a
    filter to a one-dimensional stimulus.
    """
    filt = np.random.randn(100,)
    stim = np.random.randn(1000,)
    pred = flt.linear_prediction(filt, stim)

    sl = slicestim(stim, filt.shape[0])
    assert np.allclose(filt.reshape(1, -1).dot(sl), pred)


def test_linear_prediction_multi_dim():
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


def test_revco():
    """Test computation of a linear filter by reverse correlation"""
    # Create fake filter
    filter_length = 100
    x = np.linspace(0, 2 * np.pi, filter_length)
    true = np.exp(-1. * x) * np.sin(x)
    true /= np.linalg.norm(true)

    # Compute linear response
    stim_length = 10000
    stimulus = np.random.randn(stim_length,)
    response = np.convolve(stimulus, true, mode='full')[-stimulus.size:]

    # Reverse correlation
    filt = flt.revco(response, stimulus, filter_length, norm=True)
    tol = 0.1
    assert np.allclose(true, filt, atol=tol)
