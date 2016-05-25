"""
pyret utilities
"""

from functools import wraps
import matplotlib.pyplot as plt


def plotwrapper(func):
    """Decorator that adds axis and figure keyword arguments to the kwargs
    of a function"""

    @wraps(func)
    def wrapper(*args, figsize=None, **kwargs):

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
