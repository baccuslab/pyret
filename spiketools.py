"""
Tools for spike train analysis

Includes an object class, SpikingEvent, that is useful for detecting and analyzing firing events within
a spike raster. Also provides functions for binning spike times into a histogram (`binspikes`) and a function
for smoothing a histogram into a firing rate (`estfr`)

"""

import numpy as _np
import matplotlib.pyplot as _plt
from .peakdetect import peakdet

def binspikes(spk, tmax=None, binsize=0.01, time=None, num_trials=1):
    """
    Bin spike times at the given resolution. The function has two forms.

    Parameters
    ----------
    spk : array_like
        Array of spike times

    tmax : float, optional
        Final stop time. Only takes spikes that occur before this time.
        If None (default), this is set to the final spike time

    binsize : float, optional
        Size of bins in seconds (Default: 0.01 seconds)

    time : array_like, optional
        If None (default), the `tmax` and `binsize` parameters are used to generate the time array.
        Otherwise, the given array is used as the time bins for the spike histogram

    num_trials : float, optional
        How many trials went into this binning. The output counts are normalized such that
        they represent # of spikes / trial (Default: 1)

    Returns
    ------
    bspk : array_like
        Binned spike times

    tax : array_like
        The time points corresponding to the bin centers (has the same size as `bspk`)

    """

    # if time is not specified, create a time vector
    if time is None:

        # If a max time is not specified, set it to the time of the last spike
        if not tmax:
            tmax = _np.ceil(spk.max())

        # create the time vector
        time = _np.arange(0, tmax + binsize, binsize)

    # bin spike times
    bspk = _np.histogram(spk, bins=time)[0].astype(float)

    # center the time bins
    tax = time[:-1] + 0.5 * _np.mean(_np.diff(time))

    # returned binned spikes and cenetered time axis
    return bspk / num_trials, tax


def estfr(tax, bspk, sigma=0.01):
    """
    Estimate the instantaneous firing rates from binned spike counts.

    Parameters
    ----------
    tax : array_like
        Array of time points corresponding to bins (as from binspikes)

    bspk : array_like
        Array of binned spike counts (as from binspikes)

    sigma : float, optional
        The width of the Gaussian filter, in seconds (Default: 0.01 seconds)

    Returns
    -------
    rates : array_like
        Array of estimated instantaneous firing rate

    """

    # estimate binned spikes time step
    dt = float(_np.mean(_np.diff(tax)))

    # Construct Gaussian filter, make sure it is normalized
    tau  = _np.arange(-5 * sigma, 5 * sigma, dt)
    filt = _np.exp(-0.5 * (tau / sigma) ** 2)
    filt = filt / _np.sum(filt)
    size = _np.round(filt.size / 2)

    # Filter  binned spike times
    return _np.convolve(filt, bspk, mode='full')[size:size + tax.size] / dt


class SpikingEvent(object):
    """
    The spiking event class bundles together functions that are used to analyze
    individual firing events, consisting of spiking activity recorded across trials / cells / conditions.

    Attributes
    ----------
    start : float
        the start time of the firing event

    stop : float
        the stop time of the firing event

    spikes : array_like
        the spikes associated with this firing event. This data is stored as an (n by 2) numpy array,
        where the first column is the set of spike times in the event and the second column is a list of
        corresponding trial/cell/condition indices for each spike

    """

    def __init__(self, start_time, stop_time, spikes):
        self.start = start_time
        self.stop = stop_time
        self.spikes = spikes

    def __str__(self):
        """
        Printing this object prints out the start / stop time and number of spikes in the event

        """
        return '%5.2fs - %5.2fs (%i spikes)' % (self.start, self.stop, self.spikes.shape[0])

    def __eq__(self, other):
        """
        Equality between two spiking events is true if the start & stop times are the same

        """
        return (self.start == other.start) & (self.stop == other.stop)

    def trial_counts(self):
        """
        Count the number of spikes per trial

        Usage
        -----
        counts = spkevent.trial_counts()

        """
        counts, _ = _np.histogram(self.spikes[:,1], bins=_np.arange(_np.min(self.spikes[:,1]),
                                                                    _np.max(self.spikes[:,1])))
        return counts

    def event_stats(self):
        """
        Compute statistics (mean and standard deviation) across trial spike counts

        Usage
        -----
        mu, sigma = spkevent.event_stats()

        """

        # count number of spikes per trial
        counts = self.trial_counts()

        return _np.mean(counts), _np.std(counts)

    def ttfs(self):
        """
        Computes the time to first spike for each trial, ignoring trials that had zero spikes

        Usage
        -----
        times = spkevent.ttfs()

        """
        trials, indices = _np.unique(self.spikes[:,1], return_index=True)[:2]
        return self.spikes[indices,0]
    
    def jitter(self):
        """
        Computes the jitter (standard deviation) in the time to first spike across trials

        Usage
        -----
        sigma = spkevent.jitter()

        """
        return _np.std(self.ttfs())

    def sort(self):
        """
        Sort trial indices by the time to first spike

        Usage
        -----
        sortedspikes = spkevent.sort()

        """

        # get first spike in each trial
        trial_indices = _np.unique(self.spikes[:,1], return_index=True)[1]

        # sort by time of first spike
        sorted_indices = _np.argsort(self.spikes[trial_indices, 0])

        # get reassigned trials
        sorted_trials = self.spikes[trial_indices[sorted_indices], 1]

        # store new spiking array, resetting trial numbers to the new index values
        sortedspikes = self.spikes.copy()
        for idx in range(sorted_trials.size):
            sortedspikes[self.spikes[:,1] == sorted_trials[idx],1] = idx + 1

        return sortedspikes

    def plot(self, sort=False, ax=None, color='SlateGray'):
        """
        Plots this event, as a spike raster

        Usage
        -----
        spkevent.plot()

        Parameters
        ----------
        sort : boolean, optional
            Whether or not to sort the raster by the time to first spike (Default: False)

        ax : matplotlib Axes object, optional
            If given, plots the raster on these axes. If None, creates a new figure (Default: None)

        color : string
            The color of the points in the raster (Default: 'SlateGray')

        """

        if sort:
            spikes = self.sort()
        else:
            spikes = self.spikes

        if not ax:
            ax = _plt.figure().add_subplot(111)

        ax.plot(spikes[:,0], spikes[:,1], 'o', markersize=6, markerfacecolor=color)


def detectevents(spk, threshold=(0.3,0.05)):
    """
    Detects spiking events given a PSTH and spike times for multiple trials

    Usage
    -----
    events = detectevents(spikes, threshold=(0.1, 0.005))

    Parameters
    ----------
    spk : array_like
        An (n by 2) array of spike times, indexed by trial / condition.
        The first column is the set of spike times in the event and the second column is a list of
        corresponding trial/cell/condition indices for each spike.

    threshold : (float, float), optional
        A tuple of two floats that are used as thresholds for detecting firing events. Default: (0.1, 0.005)
        See `peakdetect.py` for more info

    Returns
    -------
    events : list
        A list of 'spikingevent' objects, one for each firing event detected.
        See the `spikingevent` class for more info.

    """

    # find peaks in the PSTH
    bspk, tax = binspikes(spk[:,0], binsize=0.01, num_trials=_np.max(spk[:,1]))
    psth      = estfr(tax, bspk, sigma=0.005)
    maxtab, _ = peakdet(psth, threshold[0], tax)

    # store spiking events in a list
    events = list()

    # join similar peaks, define events
    for eventidx in range(maxtab.shape[0]):

        # get putative start and stop indices of each spiking event, based on the firing rate
        start_indices = _np.where( (psth <= threshold[1]) & (tax < maxtab[eventidx,0]) )[0]
        stop_indices = _np.where( (psth <= threshold[1]) & (tax > maxtab[eventidx,0]) )[0]

        # find the start time, defined as the right most peak index
        starttime = tax[0] if start_indices.size == 0 else tax[_np.max(start_indices)]

        # find the stop time, defined as the lest most peak index
        stoptime = tax[-1] if stop_indices.size == 0 else tax[_np.min(stop_indices )]

        # find spikes within this time interval (these make up the spiking event)
        event_spikes = spk[(spk[:,0] >= starttime) & (spk[:,0] < stoptime),:]

        # create the spiking event
        event = SpikingEvent(starttime, stoptime, event_spikes)

        # only add it if it is a unique event
        if not events or not (events[-1] == event):
            events.append(event)

    return tax, psth, bspk, events
