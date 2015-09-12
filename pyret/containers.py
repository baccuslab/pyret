"""
Objects for holding relevant experimental data
"""

from .filtertools import rolling_window
from .spiketools import binspikes
from . import visualizations as viz
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
        print('Spikes for {:d} cells'.format(len(cells)))
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

    def stim_sliced(self, history, batch_size=-1):
        """Returns a view into the stimulus array"""

        sliced_array = np.rollaxis(rolling_window(self.stim, history), 3, 2)

        if batch_size > 0:
            return partition_last(sliced_array, batch_size)
        else:
            return sliced_array

    def spike_history(self, history, offset=1, batch_size=-1):
        """Returns a view into the spikes array, offset by some amount"""
        arr = np.hstack((np.zeros((self.ncells, offset)),
                         self.spikes[:, offset:]))
        sliced_array = np.rollaxis(rolling_window(arr, history), 2, 1)

        if batch_size > 0:
            return partition_last(sliced_array, batch_size)
        else:
            return sliced_array

    def ste(self, stim_hist, ci):
        return (self.stim[..., (t-stim_hist):t].astype('float')
                for t in range(len(self))
                if self.spikes[ci, t] > 0 and t >= stim_hist)


class Filter(np.ndarray):
    """
    Container for a spatiotemporal or temporal filter
    """

    def __new__(cls, arr, dt, dx=None, dy=None, tstart=0.):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(arr).view(cls)

        # add the spatial and temporal resolutions
        obj.dt = dt
        obj.dx = dx
        obj.dy = dy

        # time at which the STA starts
        obj.tstart = tstart

        # time array corresponding to this filter
        obj.tax = np.linspace(obj.tstart, -dt*arr.shape[-1], arr.shape[-1])

        # Finally, we must return the newly created object:
        return obj

    @property
    def length(self):
        return len(self.tax) * self.dt

    def __str__(self):
        return '{}s filter with {} spatial dimensions'.format(self.length,
                                                              self.shape[1:])

    def plot(self):
        viz.plotsta(self.tax, self)

    def play(self):
        viz.playsta(self)
