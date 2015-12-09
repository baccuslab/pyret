"""
Containers for holding relevant experimental data

"""

from . import filtertools as ft
from .spiketools import binspikes
from . import visualizations as viz
import numpy as np

__all__ = ['Experiment', 'Filter']


class Experiment(object):
        """
        Sensory experiment data container
        """

        def __init__(self, stimulus, stim_time, spikes, dt):
            """

            Parameters
            ----------
            stimulus : array_like

            stim_time : array_like

            spikes : list of array_like

            dt : float
                Temporal sampling rate (seconds)

            """

            assert type(stimulus) is np.ndarray, \
                "Stimulus must be a numpy array"

            assert type(stim_time) == np.ndarray and stim_time.ndim == 1, \
                "Time vector must be a 1-D numpy array"

            assert len(stim_time) == len(stimulus), \
                "The stimulus time array must have the same length as the stimulus"

            self.stim = stimulus
            self.time = stim_time
            self.spikes = spikes
            self.dt = dt

        def __len__(self):
            return len(self.time)

        def sta(self, cellidx, history):
            """
            Returns the STA for the given cell as a Filter object

            Parameters
            ----------
            cellidx : int
                The index of the cell to analyze

            history : int
                Number of samples of temporal history to include

            """

            sta = ft.getsta(self.time, self.stim, self.spikes[cellidx], history)[0]
            return Filter(np.flipud(sta), self.dt)

        def binspikes(self, cellidx):
            """
            Bin spike times for the given cell

            Parameters
            ----------
            cellidx : int
                Index of the cell

            """

            spk = self.spikes[cellidx]
            return np.append(0, binspikes(spk, time=self.time)[0])

        @property
        def length(self):
            return self.time[-1]

        @property
        def ncells(self):
            return len(self.spikes)

        def stim_sliced(self, history):
            """Returns a view into the stimulus array"""
            return ft.rolling_window(self.stim, history, time_axis=0)


class Filter:
    """
    Container for a spatiotemporal or temporal filter

    """

    def __init__(self, data, dt, dx=None, dy=None):
        """
        Creates a spatiotemporal filter object

        Parameters
        ----------
        data : array_like
            The spatiotemporal filter, either (time,) or (time, space) or
            (time, space, space) depending on the number of spatial dimensions

        dt : float

        dx : float, optional
        dy : float, optional

        """

        assert type(data) is np.ndarray, "Filter data must be a numpy ndarray"

        # store the filter values
        self.data = data

        # sampling rates in time and each spatial dimension
        self.dt = dt
        self.dx = dx
        self.dy = dy

    @property
    def spatial(self):
        """
        Gets the spatial RF
        """
        return ft.decompose(self.data)[0]

    @property
    def temporal(self):
        """
        Gets the spatial RF
        """
        return ft.decompose(self.data)[1]

    def plot(self):
        """
        Plots the spatiotemporal filter
        """
        viz.plotsta(self.tax, self.data)

    def play(self):
        """
        Plays a movie of the spatiotemporal filter
        """
        assert self.ndim >= 2, "playsta only valid for spatiotemporal stimuli"
        viz.playsta(self.data)

    def __repr__(self):
        return repr(self.data)

    def __str__(self):
        return "{:0.0f} ms long filter with shape: {}" \
            .format(1000 * self.length, self.shape)

    def __len__(self):
        """
        Returns the length of the filter in samples
        """
        return len(self.data)

    def __iter__(self):
        """
        Returns an iterator over the temporal frames in the filter
        """
        return iter(self.data)

    @property
    def length(self):
        """
        Returns the filter length in seconds
        """
        return len(self) * self.dt

    @property
    def shape(self):
        """
        Returns the shape of the filter
        """
        return self.data.shape

    @property
    def ndim(self):
        """
        Returns the number of filter dimensions
        """
        return self.data.ndim

    @property
    def tax(self):
        """
        Creates a temporal axis for this filter
        """
        return np.arange(0, -self.length, -self.dt)
