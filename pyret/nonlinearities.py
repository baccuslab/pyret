"""
Tools for fitting nonlinear functions to data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from functools import wraps
from itertools import zip_longest

__all__ = ['Sigmoid', 'Binterp']


class Nonlinearity:
    def plot(self, span=(-5, 5), n=100):
        """Creates a 1D plot of the nonlinearity"""
        x = np.linspace(span[0], span[1], n)
        plt.plot(x, self.predict(x))

    def fit(self, x, y):
        """Fits the parameters of the nonlinearity

        Parameters
        ----------
        x : array_like
            input to the nonlinearity

        y : array_like
            output of the nonlinearity (must have the same shape as x)
        """
        raise NotImplementedError

    def predict(self, x):
        """Computes the value of the function at the given input

        Parameters
        ----------
        x : array_like
            The input to the nonlinearity

        Returns
        y : array_like
            The output of the nonlinearity
        """
        raise NotImplementedError

    @wraps(predict)
    def __call__(self, x):
        return self.predict(x)


class Sigmoid(Nonlinearity):
    def __init__(self, baseline=0., peak=1., slope=1., threshold=0.):
        """A sigmoidal nonlinearity

        Estimates a nonlinearity of the following form:
        .. math:: f(x) = \beta + \frac{\alpha}{(1 + \exp(-\gamma * (x - \theta)))}

        Usage
        -----
        >>> f = Sigmoid().fit(x_train, y_train)
        >>> yhat = f.predict(x_test)        # f(x_test) works as well

        Parameters
        ----------
        baseline : float
            y-offset (baseline)

        peak : float
            maximum response

        slope : float
            gain of the sigmoid

        threshold : float
            midpoint of the sigmoid
        """
        self.init_params = (baseline, peak, slope, threshold)

    def fit(self, x, y, **kwargs):
        self.params, self.pcov = curve_fit(self._sigmoid, x, y, self.init_params, **kwargs)
        return self

    @staticmethod
    def _sigmoid(x, threshold, slope, peak, baseline):
        return baseline + peak / (1 + np.exp(-slope * (x - threshold)))

    def predict(self, x):
        try:
            return self._sigmoid(x, *self.params)
        except NameError:
            raise RuntimeError('No estimated parameters, call fit() first')


class Binterp(Nonlinearity):
    def __init__(self, nbins, method='linear', fill_value='extrapolate'):
        """Interpolated nonlinearity by sorting and binning the data

        Given samples (x, y) from the nonlinearity, bin the values using
        variable-sized bins with roughly equal counts, and then interpolates
        between the mean y-value in each bin using scipy.interpolate.interp1d.

        Parameters
        ----------
        nbins : int
            How many bins to use along the input axis

        method : str, optional
            How to do the interpolation (Default: 'linear'). Possible values: 'linear',
            'quadratic', 'cubic', 'nearest', 'slinear', 'zero'. See scipy.interpolate.interp1d
            for details.

        fill_value : str or value, optional
            How to fill in values outside the range of bins (Default: 'extrapolate')
            Note: 'extrapolate' only works for the 'linear' or 'nearest' methods,
            see scipy.interpolate.interp1d for details
        """
        self.nbins = nbins
        self.method = method
        self.fill_value = fill_value

    @staticmethod
    def _grouper(iterable, n, fillvalue=None):
        """Collect data into fixed-length chunks or blocks"""
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n

        # TODO: make this more performant
        return np.array(list(zip_longest(*args, fillvalue=fillvalue)))

    def fit(self, x, y):
        binsize = int(x.size / self.nbins)

        # sort the x-values and create variable bin edges with equal counts
        indices = np.argsort(x)
        self.bins = x[indices][::binsize]
        y_grouped = self._grouper(y[indices], binsize, fillvalue=np.nan)
        self.values = np.nanmean(y_grouped, axis=1)

        # set the predict function using scipy.interpolate.interp1d
        self.predict = interp1d(self.bins, self.values, kind=self.method, fill_value=self.fill_value)
        return self

    def predict(self, x):
        """Placeholder, this method gets overwritten when fit() is called"""
        raise RuntimeError('No estimated parameters, call fit() first')
