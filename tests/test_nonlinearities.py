"""test_nonlinearities.py
Test code for pyret's nonlinearities module.
(C) 2016 The Baccus Lab
"""
import numpy as np

from pyret import nonlinearities

def test_sigmoid():
    """Test the Sigmoid nonlinearity class"""
    # True parameters
    thresh = 0.5
    slope = 2
    peak = 1.5
    baseline = 0.2
    n = 1000        # Number of simulate data points
    xscale = 2      # Scale factor for input range
    noise = 0.1     # Standard deviation of AWGN

    # Simulate data
    x = np.random.randn(n,) * xscale
    y = nonlinearities.Sigmoid._sigmoid(x, thresh, slope,
            peak, baseline) + np.random.randn(n,) * noise

    # Fit nonlinearity and compare
    y_hat = nonlinearities.Sigmoid().fit(x, y).predict(x)
    norm = (np.linalg.norm(y - y_hat) / np.linalg.norm(y))
    if (norm > (noise * 1.5)):
        raise AssertionError("Fitting a Sigmoid nonlinearity seems " + 
                "to have failed, relative error = {0:#0.3f}".format(norm))

def test_binterp():
    """Test the Binterp nonlinearity class"""
    # True parameters
    thresh = 0.5
    slope = 2
    peak = 1.5
    baseline = 0.2
    n = 1000        # Number of simulate data points
    xscale = 2      # Scale factor for input range
    noise = 0.1     # Standard deviation of AWGN
    nbins = 25      # Number of bins in the Binterp nonlienarity

    # Simulate data
    x = np.random.randn(n,) * xscale
    y = nonlinearities.Sigmoid._sigmoid(x, thresh, slope,
            peak, baseline) + np.random.randn(n,) * noise

    # Fit nonlinearity and compare
    y_hat = nonlinearities.Binterp(nbins).fit(x, y).predict(x)
    norm = (np.linalg.norm(y - y_hat) / np.linalg.norm(y))
    if (norm > (noise * 1.5)):
        raise AssertionError("Fitting a Sigmoid nonlinearity seems " + 
                "to have failed, relative error = {0:#0.3f}".format(norm))

