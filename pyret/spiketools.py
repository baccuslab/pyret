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

__all__ = ['binspikes', 'estfr', 'sample', 'detectevents', 'peakdet',
           'split_trials', 'SpikingEvent']


def split_trials(spikes, trial_length=None, fig=None):
    """
    Plot a raster of spike times from an array of spike times

    Notes
    -----
    The `triallength` keyword specifies the length of time for each trial, and the
    `spikes` array is split up into segments of that length. These groups are then
    plotted on top of one another, as individual trials.

    Parameters
    ----------
    spikes : array_like
        An array of spike times

    triallength : float
        The length of each trial to stack, in seconds.

    Returns
    -------
    spiketimes : array_like
        An array of spike times relative to the start of each trial.

    trials : array_like
        An array of labels corresponding to the trial associated with each
        spike in the spiketimes array.

    """

    # Compute a trial index for each spike
    trials = map(lambda s: np.floor(s, trial_length) + 1, spikes)

    return np.mod(spikes, trial_length), np.array(list(trials))


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
        If None (default), the `tmax` and `binsize` parameters are used to
        generate the time array. Otherwise, the given array is used as the time
        bins for the spike histogram.

    num_trials : float, optional
        How many trials went into this binning. The output counts are
        normalized such that they represent # of spikes / trial (Default: 1)

    Returns
    ------
    bspk : array_like
        Binned spike times

    tax : array_like
        The time points corresponding to the bin centers (same size as `bspk`)

    """

    # if time is not specified, create a time vector
    if time is None:

        # If a max time is not specified, set it to the time of the last spike
        if not tmax:
            tmax = np.ceil(spk.max())

        # create the time vector
        time = np.arange(0, tmax + binsize, binsize)

    # bin spike times
    bspk = np.histogram(spk, bins=time)[0].astype(float)

    # center the time bins
    tax = time[:-1] + 0.5 * np.mean(np.diff(time))

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
    dt = float(np.mean(np.diff(tax)))

    # Construct Gaussian filter, make sure it is normalized
    tau = np.arange(-5 * sigma, 5 * sigma, dt)
    filt = np.exp(-0.5 * (tau / sigma) ** 2)
    filt = filt / np.sum(filt)
    size = np.round(filt.size / 2)

    # Filter  binned spike times
    return np.convolve(filt, bspk, mode='full')[size:size + tax.size] / dt


def sample(rate, dt=1.0, num_trials=1):
    """
    Sample discrete spikes from a given firing rate

    Draws spikes from a Poisson distribution with mean given by `rate`

    Parameters
    ----------
    rate : array_like
        The time-varying firing rate that is the mean of the Poisson process

    dt : float, optional
        The bin size of the firing rate, in seconds (Default: 1s)

    num_trials : int, optional
        The number of trials (repeats) to draw samples for (Default: 1)

    Returns
    -------
    spikes : array_like
        An array of shape `num_trials` by `rate.shape` that contains the
        sampled number of spikes for each trial in the `rate` array

    Notes
    -----
    Spikes are drawn according to the Poisson distribution:

    .. math::

        p(n) = (\exp(-r)(r)^n) / n!

    """

    return np.random.poisson(rate * dt, (num_trials,) + rate.shape)


class SpikingEvent(object):
    """
    The spiking event class bundles together functions that are used to analyze
    individual firing events, consisting of spiking activity recorded across
    trials / cells / conditions.

    Attributes
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

    def __init__(self, start_time, stop_time, spikes):
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
        """
        Count the number of spikes per trial

        >> counts = spkevent.trial_counts()

        """
        counts, _ = np.histogram(self.spikes[:, 1], bins=np.arange(
            np.min(self.spikes[:, 1]), np.max(self.spikes[:, 1])))
        return counts

    def event_stats(self):
        """
        Compute statistics (mean and standard deviation) across spike counts

        >> mu, sigma = spkevent.event_stats()

        """

        # count number of spikes per trial
        counts = self.trial_counts()

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
        events. Default: (0.1, 0.005) see `peakdetect.py` for more info

    Returns
    -------
    events : list
        A list of 'spikingevent' objects, one for each firing event detected.
        See the `spikingevent` class for more info.

    """

    # find peaks in the PSTH
    bspk, tax = binspikes(spk[:, 0], binsize=0.01,
                          num_trials=np.max(spk[:, 1]))
    psth = estfr(tax, bspk, sigma=0.02)
    maxtab, _ = peakdet(psth, threshold[0], tax)

    # store spiking events in a list
    events = list()

    # join similar peaks, define events
    for eventidx in range(maxtab.shape[0]):

        # get putative start and stop indices of each spiking event
        start_indices = np.where((psth <= threshold[1]) &
                                  (tax < maxtab[eventidx, 0]))[0]
        stop_indices = np.where((psth <= threshold[1]) &
                                 (tax > maxtab[eventidx, 0]))[0]

        # find the start time, defined as the right most peak index
        if start_indices.size == 0:
            starttime = tax[0]
        else:
            starttime = tax[np.max(start_indices)]

        # find the stop time, defined as the lest most peak index
        if stop_indices.size == 0:
            stoptime = tax[-1]
        else:
            stoptime = tax[np.min(stop_indices)]

        # find spikes within this time interval
        event_spikes = spk[(spk[:, 0] >= starttime) &
                           (spk[:, 0] < stoptime), :]

        # create the spiking event
        event = SpikingEvent(starttime, stoptime, event_spikes)

        # only add it if it is a unique event
        if not events or not (events[-1] == event):
            events.append(event)

    return tax, psth, bspk, events


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
