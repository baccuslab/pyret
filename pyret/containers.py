"""
Objects for holding relevant experimental data
"""

from .filtertools import rolling_window
from .spiketools import binspikes
import numpy as np

__all__ = ['Experiment', 'Filter']


class Experiment(object):
    """
    Sensory experiment data container
    """

    def __init__(self, stim, time, cells, dt):
        """
        TODO: Docstring for __init__.

        Parameters
        ----------
        stim : TODO
        time : TODO
        spikes : TODO

        Returns
        -------
        TODO

        """

        assert type(stim) == np.ndarray and stim.ndim == 3, \
            "Stimulus must be a 3 dimensionsal (space x space x time) array"

        assert type(time) == np.ndarray and time.ndim == 1, \
            "Time vector must be an one dimensional numpy array"

        self.stim = stim
        self.cells = cells
        self.time = time
        self.dt = dt

        spikes = list()
        for cell in self.cells:
            spikes.append(np.append(0, binspikes(cell, time=time)[0]))
        self.spikes = np.vstack(spikes)

    def __len__(self):
        return len(self.time)

    @property
    def tmax(self):
        return self.time[-1]

    @property
    def ncells(self):
        return self.spikes.shape[0]

    def stim_sliced(self, history):
        """Returns a view into the stimulus array"""
        return np.rollaxis(rolling_window(self.stim, history), 3, 2)

    def spike_history(self, history, offset=1):
        """Returns a view into the spikes array, offset by some amount"""
        arr = np.hstack((np.zeros((self.ncells,offset)), self.spikes[:, offset:]))
        return np.rollaxis(rolling_window(arr, history), 2, 1)

    def ste(self, stim_hist, ci):
        return (self.stim[..., (t-stim_hist):t].astype('float')
                for t in range(len(self))
                if self.spikes[ci,t] > 0 and t >= stim_hist)


class Filter(np.ndarray):
    """
    Container for a spatiotemporal or temporal filter
    """

    def __new__(cls, input_array, dt):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # add the new attribute to the created instance
        obj.dt = dt
        obj.tau = np.linspace(0, -dt*input_array.shape[-1], input_array.shape[-1])

        # Finally, we must return the newly created object:
        return obj

    # def __array_finalize__(self, obj):
        # if obj is None: return
        # self.info = getattr(obj, 'info', 1)
