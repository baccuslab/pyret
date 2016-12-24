"""
pyret utilities
"""

from functools import wraps
import matplotlib.pyplot as plt


def flat2d(x):
    """Flattens all dimensions after the first of the given array

    Useful for collapsing spatial dimensions in a spatiotemporal
    stimulus or filter.
    """
    return x.reshape(x.shape[0], -1)


def plotwrapper(func):
    """Decorator that adds axis and figure keyword arguments to the kwargs
    of a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        figsize = kwargs.get('figsize', None)

        if 'ax' not in kwargs:
            if 'fig' not in kwargs:
                kwargs['fig'] = plt.figure(figsize=figsize)
            kwargs['ax'] = kwargs['fig'].add_subplot(111)
        else:
            if 'fig' not in kwargs:
                kwargs['fig'] = kwargs['ax'].get_figure()

        func(*args, **kwargs)
        plt.show()
        plt.draw()
        return kwargs['fig'], kwargs['ax']

    return wrapper
