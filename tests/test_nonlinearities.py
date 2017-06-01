"""test_nonlinearities.py
Test code for pyret's nonlinearities module.
(C) 2016 The Baccus Lab
"""
import numpy as np
import pytest
import pyret.nonlinearities as nln
import sklearn


nonlinearities = [
    (nln.Sigmoid, (), 1234, 0.1, 0.99),
    (nln.Binterp, (50,), 1234, 0.1, 0.99),
    (nln.RBF, (11,), 1234, 0.1, 0.99),
    (nln.GaussianProcess, (), 1234, 0.1, 0.99),
    (nln.Sigmoid, (), 5678, 0.5, 0.99),
    (nln.Binterp, (25,), 5678, 0.5, 0.99),
    (nln.RBF, (11,), 1234, 0.5, 0.99),
    (nln.GaussianProcess, (), 5678, 0.5, 0.97),
]


@pytest.mark.parametrize("nln_cls", [nln.Sigmoid, nln.Binterp, nln.GaussianProcess])
def test_inheritance(nln_cls):
    assert issubclass(nln_cls, sklearn.base.BaseEstimator)
    assert issubclass(nln_cls, sklearn.base.RegressorMixin)
    assert issubclass(nln_cls, nln.NonlinearityMixin)


@pytest.mark.parametrize("nln_cls,args,seed,noise_stdev,r2_thresh", nonlinearities)
def test_exception(nln_cls, args, seed, noise_stdev, r2_thresh):
    with pytest.raises(sklearn.exceptions.NotFittedError):
        nln_cls(*args).predict(np.random.randn(100,))


@pytest.mark.parametrize("nln_cls,args,seed,noise_stdev,r2_thresh", nonlinearities)
def test_fitting(nln_cls, args, seed, noise_stdev, r2_thresh):
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
    model = nln_cls(*args).fit(x, y_obs)

    # compute coefficient of determination (r2)
    assert model.score(x, y) >= r2_thresh
