"""test_nonlinearities.py
Test code for pyret's nonlinearities module.
(C) 2016 The Baccus Lab
"""
import numpy as np
import pytest
import pyret.nonlinearities as nln


nonlinearities = [
    (nln.Sigmoid, (), 1234, 0.1),
    (nln.Binterp, (50,), 1234, 0.1),
    (nln.GaussianProcess, (), 1234, 0.1),
    (nln.Sigmoid, (), 5678, 0.5),
    (nln.Binterp, (25,), 5678, 0.5),
    (nln.GaussianProcess, (), 5678, 0.5),
]


@pytest.mark.parametrize("nln_cls,args,seed,noise_stdev", nonlinearities)
def test_fitting(nln_cls, args, seed, noise_stdev):
    """Test the fit method of each nonlinearity"""
    np.random.seed(seed)

    # simulate a noisy nonlinearity
    thresh = 0.5
    slope = 2
    peak = 1.5
    baseline = 0.2
    n = 1000            # Number of simulate data points
    xscale = 2          # Scale factor for input range
    x = np.random.randn(n,) * xscale
    y = nln.Sigmoid._sigmoid(x, thresh, slope, peak, baseline)
    y_obs = y + np.random.randn(n,) * noise_stdev

    # fit nonlinearity to the observed (noisy) data
    y_hat = nln_cls(*args).fit(x, y_obs).predict(x)

    # compute relative error
    rel_error = np.linalg.norm(y - y_hat) / np.linalg.norm(y)
    assert rel_error < (0.5 * noise_stdev)
