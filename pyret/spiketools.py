"""
Tools for spike train analysis

Includes an object class, SpikingEvent, that is useful for detecting and
analyzing firing events within a spike raster. Also provides functions for
binning spike times into a histogram (`binspikes`) and a function
for smoothing a histogram into a firing rate (`estfr`)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

__all__ = ['binspikes', 'estfr', 'detectevents', 'peakdet', 'SpikingEvent']


def binspikes(spk, time):
    """
    Bin spike times at the given resolution. The function has two forms.

    Parameters
    ----------
    spk : array_like
        Array of spike times

    time : array_like
        The left edges of the time bins.

    Returns
    -------
    bspk : array_like
        Binned spike times
    """
    bin_edges = np.append(time, 2 * time[-1] - time[-2])
    return np.histogram(spk, bins=bin_edges)[0].astype(float)


def estfr(bspk, time, sigma=0.01):
    """
    Estimate the instantaneous firing rates from binned spike counts.

    Parameters
    ----------
    bspk : array_like
        Array of binned spike counts (e.g. from binspikes)

    time : array_like
        Array of time points corresponding to bins

    sigma : float, optional
        The width of the Gaussian filter, in seconds (Default: 0.01 seconds)

    Returns
    -------
    rates : array_like
        Array of estimated instantaneous firing rate
    """
    # estimate the time resolution
    dt = float(np.mean(np.diff(time)))

    # Construct Gaussian filter, make sure it is normalized
    tau = np.arange(-5 * sigma, 5 * sigma, dt)
    filt = np.exp(-0.5 * (tau / sigma) ** 2)
    filt = filt / np.sum(filt)
    size = int(np.round(filt.size / 2))

    # Filter  binned spike times
    return np.convolve(filt, bspk, mode='full')[size:size + time.size] / dt


class SpikingEvent(object):
    def __init__(self, start_time, stop_time, spikes):
        """
        The spiking event class bundles together functions that are used to analyze
        individual firing events, consisting of spiking activity recorded across
        trials / cells / conditions.

        Parameters
        ----------
        start : float
            the start time of the firing event

        stop : float
            the stop time of the firing event

        spikes : array_like
            the spikes associated with this firing event. This data is stored as an
            (n by 2) numpy array, where the first column is the set of spike times
            in the event and the second column is a list of corresponding
            trial/cell/condition indices for each spike
        """
        self.start = start_time
        self.stop = stop_time
        self.spikes = spikes

    def __str__(self):
        return '%5.2fs - %5.2fs (%i spikes)' % (self.start, self.stop,
                                                self.spikes.shape[0])

    def __eq__(self, other):
        """
        Equality between two spiking events is true if the start & stop times
        are the same

        """
        return (self.start == other.start) & (self.stop == other.stop)

    def trial_counts(self):
        """Count the number of spikes per trial"""
        return Counter(self.spikes[:, 1])

    def stats(self):
        """
        Compute statistics (mean and standard deviation) across spike counts

        >> mu, sigma = spkevent.event_stats()

        """

        # count number of spikes per trial
        counts = list(self.trial_counts().values())

        return np.mean(counts), np.std(counts)

    def ttfs(self):
        """
        Computes the time to first spike for each trial, ignoring trials that
        had zero spikes

        >> times = spkevent.ttfs()

        """
        trials, indices = np.unique(self.spikes[:, 1], return_index=True)[:2]
        return self.spikes[indices, 0]

    def jitter(self):
        """
        Computes the jitter (standard deviation) in the time to first spike

        >> sigma = spkevent.jitter()

        """
        return np.std(self.ttfs())

    def sort(self):
        """
        Sort trial indices by the time to first spike

        >> sortedspikes = spkevent.sort()

        """

        # get first spike in each trial
        trial_indices = np.unique(self.spikes[:, 1], return_index=True)[1]

        # sort by time of first spike
        sorted_indices = np.argsort(self.spikes[trial_indices, 0])

        # get reassigned trials
        sorted_trials = self.spikes[trial_indices[sorted_indices], 1]

        # store new spiking array, resetting trial numbers
        sortedspikes = self.spikes.copy()
        for idx in range(sorted_trials.size):
            sortedspikes[self.spikes[:, 1] == sorted_trials[idx], 1] = idx + 1

        return sortedspikes

    def plot(self, sort=False, ax=None, color='SlateGray'):
        """
        Plots this event, as a spike raster

        >> spkevent.plot()

        Parameters
        ----------
        sort : boolean, optional
            Whether or not to sort by the time to first spike (Default: False)

        ax : matplotlib Axes object, optional
            If None, creates a new figure (Default: None)

        color : string
            The color of the points in the raster (Default: 'SlateGray')

        """

        if sort:
            spikes = self.sort()
        else:
            spikes = self.spikes

        if not ax:
            ax = plt.figure().add_subplot(111)

        ax.plot(spikes[:, 0], spikes[:, 1], 'o', markersize=6,
                markerfacecolor=color)


def detectevents(spk, threshold=(0.3, 0.05)):
    """
    Detects spiking events given a PSTH and spike times for multiple trials

    >> events = detectevents(spikes, threshold=(0.1, 0.005))

    Parameters
    ----------
    spk : array_like
        An (n by 2) array of spike times, indexed by trial / condition.
        The first column is the set of spike times in the event and the second
        column is a list of corresponding trial/cell/condition indices
        for each spike.

    threshold : (float, float), optional
        A tuple of two floats that are used as thresholds for detecting firing
        events. Default: (0.1, 0.005) see `peakdet` for more info

    Returns
    -------
    events : list
        A list of 'spikingevent' objects, one for each firing event detected.
        See the `spikingevent` class for more info.
    """
    # find peaks in the PSTH
    time = np.arange(0, np.ceil(spk[:, 0].max()), 0.01)
    bspk = binspikes(spk[:, 0], time)
    psth = estfr(bspk, time, sigma=0.01)
    maxtab, _ = peakdet(psth, threshold[0], time)

    # store spiking events in a list
    events = list()

    # join similar peaks, define events
    for eventidx in range(maxtab.shape[0]):

        # get putative start and stop indices of each spiking event
        start_indices = np.where((psth <= threshold[1]) &
                                 (time < maxtab[eventidx, 0]))[0]
        stop_indices = np.where((psth <= threshold[1]) &
                                (time > maxtab[eventidx, 0]))[0]

        # find the start time, defined as the right most peak index
        if start_indices.size == 0:
            starttime = time[0]
        else:
            starttime = time[np.max(start_indices)]

        # find the stop time, defined as the lest most peak index
        if stop_indices.size == 0:
            stoptime = time[-1]
        else:
            stoptime = time[np.min(stop_indices)]

        # find spikes within this time interval
        event_spikes = spk[(spk[:, 0] >= starttime) &
                           (spk[:, 0] < stoptime), :]

        # create the spiking event
        event = SpikingEvent(starttime, stoptime, event_spikes)

        # only add it if it is a unique event
        if not events or not (events[-1] == event):
            events.append(event)

    return time, psth, bspk, events


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays containing the maxima and minima of a 1D signal

    Parameters
    ----------
    v : array_like
        The input signal (array) to find the peaks of

    delta : float
        The threshold for peak detection. A point is considered a maxima
         (or minima) if it is at least delta larger (or smaller) than
         its neighboring points

    x : array_like, optional
        If given, the locations of the peaks are given as the corresponding
        values in `x`. Otherwise, the locations are given as indices

    Returns
    -------
    maxtab : array_like
        An (N x 2) array containing the indices or locations (left column)
        of the local maxima in `v` along with the corresponding maximum
        values (right column).

    mintab : array_like
        An (M x 2) array containing the indices or locations (left column)
        of the local minima in `v` along with the corresponding minimum
        values (right column).
    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)
