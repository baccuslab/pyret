'''
spiketools.py

Tools for basic manipulation of spike trains

(c) 2014 bnaecker, nirum
'''

import numpy as np
from scipy.io import loadmat
from scipy import signal

try:
    from peakdetect import peakdet
except ImportError:
    raise ImportError('You need to have the peakdetect module available on your python path.')

def binspikes(spk, tmax=None, binsize=0.01, time=None):
    '''
    
    Bin spike times at the given resolution. The function has two forms.

    Input
    -----

    spk:
        Array of spike times
	
    EITHER:

        tmax:
            Maximum bin time. Usually end of experiment, but could
            really be anything.

        binsize:
            Size of bins (in milliseconds).

    OR:

        time:
            The array to use as the actual bins to np.histogram

    Output
    ------

    bspk:
        Binned spike times

    tax:
        The bins themselves.

    '''

    # Check if actual time bins are specified
    if time is not None:
        return np.histogram(spk, bins=time)

    # If not, use either tmax or the maximum spike time and the binsize
    if not tmax:
        tmax = spk.max()
    tbins = np.arange(0, tmax, binsize)
    bspk, _ = np.histogram(spk, bins=tbins)

    return bspk, tbins[:-1] + 0.5*np.mean(np.diff(tbins))

def estfr(bspk, binsize=0.01, npts=7, sd=2):
    '''
    
    Estimate the instantaneous firing rates from binned spike counts

    Input
    -----
    bspk:
        Array of binned spike counts (as from binspikes)

    npts:
        Number of points in Gaussian filter used to smooth counts

    sd:
        SD (in points) of the Gaussian filter used to smooth counts

    Output
    ------

    rates:
        Array of estimated instantaneous firing rate

    '''
    # Construct Gaussian filter
    filt = signal.gaussian(npts, sd)

    # Filter  binned spike times
    return signal.lfilter(filt, 1, bspk) / binsize

class spikingevent:
    '''

    The spiking event class bundles together functions that are used to analyze
    individual firing events, consisting of spiking activity recorded across trials / cells / conditions.

    Properties
    ----------

    start:
        the start time of the firing event

    stop:
        the stop time of the firing event

    spikes:
        the spikes associated with this firing event. This data is stored as an (n by 2) numpy array,
        where the first column is the set of spike times in the event and the second column is a list of
        corresponding trial/cell/condition indices for each spike

    '''

    def __init__(self, startTime, stopTime, spikes):
        self.start = startTime
        self.stop = stopTime
        self.spikes = spikes

    def __repr__(self):
        '''
        Printing this object prints out the start / stop time and number of spikes in the event
        '''
        return ('%5.2fs - %5.2fs (%i spikes)' % (self.start, self.stop, self.spikes.shape[0]))

    def __eq__(self, other):
        '''
        Equality between two spiking events is true if the start & stop times are the same
        '''
        return (self.start == other.start) & (self.stop == other.stop)

    def trialCounts(self):
        '''
        Count the number of spikes per trial

        Usage: counts = spkevent.trialCounts()

        '''
        counts, _ = np.histogram(self.spikes[:,1], bins=np.arange(np.min(self.spikes[:,1]), np.max(self.spikes[:,1])))
        return counts

    def eventStats(self):
        '''
        Compute statistics (mean and standard deviation) across trial spike counts

        Usage: mu, sigma = spkevent.trialStats()

        '''

        # count number of spikes per trial
        counts = self.eventCounts()

        return np.mean(counts), np.std(counts)

    def ttfs(self):
        '''
        Computes the time to first spike for each trial, ignoring trials that had zero spikes

        Usage: times = spkevent.ttfs()

        '''
        (trials, indices) = np.unique(self.spikes[:,1], return_index=True)
        return self.spikes[indices,0]
    
    def jitter(self):
        '''
        Computes the jitter (standard deviation) in the time to first spike across trials

        Usage: sigma = spkevent.jitter()

        '''
        return np.std(self.ttfs())

    def sort(self):
        '''
        Sort trial indices by the time to first spike

        Usage: sortedspikes = spkevent.sort()

        '''

        # get first spike in each trial
        _, trialIndices = np.unique(self.spikes[:,1], return_index=True)

        # sort by time of first spike
        sortedIndices = np.argsort(self.spikes[trialIndices, 0])

        # get reassigned trials
        sortedtrials = self.spikes[trialIndices[sortedIndices], 1]

        # store new spiking array, resetting trial numbers to the new index values
        sortedspikes = self.spikes.copy()
        for idx in range(sortedtrials.size):
            sortedspikes[self.spikes[:,1]==sortedtrials[idx],1] = idx+1

        return sortedspikes

    def plot(self, sort=False, ax=None, color='SlateGray'):
        '''
        Plots this event, as a spike raster

        Usage: spkevent.plot()

        '''

        if sort:
            spikes = self.sort()
        else:
            spikes = self.spikes

        if not ax:
            ax = plt.figure().add_subplot(111)

        ax.plot(spikes[:,0], spikes[:,1], 'o', markersize=10, markercolor=color)

def detectevents(spk, threshold):
    '''

    Detects spiking events given a PSTH and spike times for multiple trials
    Usage: events = detectevents(spikes, threshold=(0.1, 0.005))

    Input
    -----
    spk:
        An (n by 2) array of spike times, indexed by trial / condition.
        The first column is the set of spike times in the event and the second column is a list of corresponding trial/cell/condition indices for each spike.

    Output
    ------
    events (list):
        A list of 'spikingevent' objects, one for each firing event detected.
        See the spikingevent class for more info.

    '''

    # find peaks in the PSTH
    bspk, tax      = binspikes(spk[:,0], tmax=None, binsize=0.01)  # bin spikes
    psth           = estfr(bspk, binsize=0.01)                     # smooth into a firing rate
    maxtab, mintab = peakdet(psth, threshold[0], tax)              # find peaks in firing rate

    # store spiking events in a list
    events = list()

    # join similar peaks, define events
    for eventidx in range(maxtab.shape[0]):

        # get putative start and stop indices of each spiking event, based on the firing rate
        startIndices, = np.where( (psth <= threshold[1]) & (tax < maxtab[eventidx,0]) )
        stopIndices,  = np.where( (psth <= threshold[1]) & (tax > maxtab[eventidx,0]) )

        # find the start time, defined as the right most peak index
        starttime = tax[0] if startIndices.size == 0 else tax[np.max(startIndices)]

        # find the stop time, defined as the lest most peak index
        stoptime = tax[-1] if  stopIndices.size == 0 else tax[np.min(stopIndices )]

        # find spikes within this time interval (these make up the spiking event)
        eventSpikes = spk[(spk[:,0] >= starttime) & (spk[:,0] < stoptime),:]

        # create the spiking event
        myEvent = spikingevent(starttime, stoptime, eventSpikes)

        # only add it if it is a unique event
        if not events or not (events[-1] == myEvent):
            events.append(myEvent)

    return events
