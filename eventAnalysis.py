import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.peakdetect import peakdet
from helpers.psth import computePSTH

"""
firing event analysis tools
author: Niru Maheswaranathan
5:13 AM Jan 4, 2013
"""

class spikingEvent:
    'spiking event'

    def __init__(self, startTime, stopTime, spikes):
        self.start = startTime
        self.stop = stopTime
        self.spikes = spikes

    def __repr__(self):
        return ('%5.2fs - %5.2fs (%i spikes)' % (self.start, self.stop, self.spikes.shape[0]))

    def __eq__(self, other):
        return (self.start == other.start) & (self.stop == other.stop)

    def eventCounts(self, numTrials):
        'number of spikes per event'
        counts, bins = np.histogram(self.spikes[:,1], bins=np.arange(numTrials[0], numTrials[1]))
        return counts

    def eventStats(self, numTrials, ignoreFirstTrial=0):
        'mean and std. dev. of event spike counts'
        counts = self.eventCounts(numTrials)

        # if the user wants, don't count the first trial (due to adaptation)
        if ignoreFirstTrial:
            counts = counts[1:]

        return np.mean(counts), np.std(counts)

    def ttfs(self):
        'time to first spike'
        (trials, indices) = np.unique(self.spikes[:,1], return_index=True)
        return self.spikes[indices,0]
    
    def jitter(self):
        'jitter in the first spike time'
        return np.std(self.ttfs())

    def sort(self):
        'returns indices sorted by the time to first spike'

        # get first spike in each trial
        _, trialIndices = np.unique(self.spikes[:,1], return_index=True)

        # sort by time of first spike
        sortedIndices = np.argsort(self.spikes[trialIndices, 0])

        # get reassigned trials
        newTrials = self.spikes[trialIndices[sortedIndices], 1]

        # store new spiking array, resetting trial numbers to the new index values
        newSpikes = self.spikes.copy()
        for idx in range(newTrials.size):
            newSpikes[self.spikes[:,1]==newTrials[idx],1] = idx+1

        return newSpikes

    def plot(self, sort=False):
        'plots this spiking event'

        if sort:
            spikes = self.sort()
        else:
            spikes = self.spikes

        plt.figure()
        plt.plot(spikes[:,0], spikes[:,1], 'ko', markersize=10)

def plotEvents(cellEvents, cellIdx, eventIdx, sort=False):
    """
    plots a raster of spikes during a spiking event
    (I'm not currently aligning events across conditions correctly)
    """

    # get number of conditions
    numConditions = len(cellEvents[0])
    
    # create the plot
    fig = plt.figure()
    fig.clf()

    # for each condition
    colors = sns.color_palette("Reds", numConditions)
    ax = list()
    for idx in range(numConditions):

        # create the subplot
        ax.append(fig.add_subplot(numConditions, 1, numConditions-idx))

        # get the spiking data
        event = cellEvents[cellIdx][idx]['events'][eventIdx]

        if sort:
            spikes = event.sort()
        else:
            spikes = event.spikes

        # add to the plot
        ax[-1].plot(spikes[:,0]-event.start, spikes[:,1], 'o', color=colors[idx], markersize=5)
        ax[-1].set_ylabel('%i' % idx)

    return fig, ax

def aggregateStats(cellEvents):

    # store statistics for each cell/condition
    meanJitter   = np.zeros((len(cellEvents),len(cellEvents[0])))
    stdJitter    = np.zeros(meanJitter.shape)
    meanSpkCount = np.zeros(meanJitter.shape)
    varSpkCount  = np.zeros(meanJitter.shape)

    # aggregate statistics
    spkCounts = list()
    varCounts = list()
    totalJitter = list()

    # for each cell
    for cidx in range(len(cellEvents)):

        # for each condition
        for eidx in range(len(cellEvents[cidx])):

            # get out events
            events = cellEvents[cidx][eidx]['events']

            # compute spike jitter
            jitter = [e.jitter() for e in events]

            # compute spike counts for each event
            exptSpkCounts = [np.mean(e.eventCounts((2,21))) for e in events]
            exptVarCounts = [ np.var(e.eventCounts((2,21))) for e in events]

            # store results
            meanJitter[cidx,eidx]   = np.mean(jitter)
            stdJitter[cidx,eidx]    = np.std(jitter)
            meanSpkCount[cidx,eidx] = np.mean(exptSpkCounts)
            varSpkCount[cidx,eidx]  = np.mean(exptVarCounts)

            # append to aggregate
            spkCounts.extend(exptSpkCounts)
            varCounts.extend(exptVarCounts)

    return meanJitter, stdJitter, meanSpkCount, varSpkCount, spkCounts, varCounts

def detectEvents(spk, centeredBins, psth, probThreshold=0.1, minProb=0.005):
    """
    " Detects spiking events in a raster "
    """

    # event detection parameters
    maxtab, mintab = peakdet(psth, probThreshold, centeredBins)

    # join similar peaks, define events
    events = list()
    for eidx in range(maxtab.shape[0]):

        # get start and stop times
        startIndices, = np.where( (psth <= minProb) & (centeredBins < maxtab[eidx,0]) )

        # if we are at the far left, start at the first bin
        if startIndices.size==0:
            startTime = centeredBins[0]

        # otherwise, start at the last bin that meets the conditions
        else:
            startTime = centeredBins[np.max(startIndices)]

        stopIndices, = np.where( (psth <= minProb) & (centeredBins > maxtab[eidx,0]) )

        # if we are at the far right, stop at the last bin
        if stopIndices.size==0:
            stopTime = centeredBins[-1]

        # otherwise, start at the first bin that meets the stop conditions
        else:
            stopTime = centeredBins[np.min(stopIndices)]

        # find spikes within this time interval
        eventSpikes = spk[(spk[:,0] >= startTime) & (spk[:,0] < stopTime),:]

        # create the spiking event
        myEvent = spikingEvent(startTime, stopTime, eventSpikes)

        # only add it if it is a unique event
        if not events or not (events[-1] == myEvent):
            events.append(myEvent)

    return events

def plotRaster(events, centeredBins, psth):

    # create figure
    fig = plt.figure()
    fig.clf()
    sns.set(style="whitegrid")

    ax = fig.add_subplot(111)

    # gray psth
    ax.plot(centeredBins, psth, linewidth=1, color='gray')

    # empty spikes
    ax.plot(spk[:,0], spk[:,1], 'o', color='gray', markersize=5)

    # event spikes
    colors = sns.color_palette("Set2", len(events))
    for eidx in range(len(events)):
        ax.plot(events[eidx].spikes[:,0], events[eidx].spikes[:,1], linestyle='None', marker='d', color=colors[eidx], markersize=5)

    ax.axis((0,30,1.5,19.5))
    plt.draw()
    plt.show()

    return fig, ax

def  getCellEvents(spikes, numCells, numExpts, binWidth=10e-3, probThreshold=0.1, minProb=0.005):

    # store events
    cellEvents = list()

    # for each cell
    for cidx in range(numCells):

        # list for this cell
        cellData = list()

        # for each condition/experiment
        for exptidx in range(numExpts):

            # get data for this cell/experiment
            spk = spikes['cells'][cidx][exptidx]

            # compute PSTH
            centeredBins, densityPSTH = computePSTH(spk, binWidth=binWidth, lpfFreq=45, density=True)
            centeredBins, psth        = computePSTH(spk, binWidth=binWidth, lpfFreq=15)

            # detect spike rasters
            cellData.append({
                'bins': centeredBins,
                'psth': psth,
                'events': detectEvents(spk, centeredBins, densityPSTH, probThreshold=probThreshold, minProb=minProb)
            })

            # plot raster
            #fig, ax = plotRaster(events, centeredBins, psth)

        cellEvents.append(cellData)

if __name__=="__main__":

    print 'Running event detection...'

    # experiment to analyze
    baseDir = '121613'
    #baseDir = '121713'

    # parameters
    binWidth = 10e-3                # bin width for computing the PSTH (seconds)
    probThreshold = 0.1             # probability of firing threshold for event detection
    minProb = 0.005                 # probability needed to separate events

    # load data
    spikes = np.load('../../data/' + baseDir + '/spikeTimes.npz')

    # info
    numCells = spikes['cells'].shape[0]
    numExpts = spikes['cells'].shape[1]

    # experiment specific stuff
    if baseDir=='121713':
        # contrasts
        baseContrast = 0.3
        epsilon = np.arange(0,0.16,0.03)

    elif baseDir=='121613':
        baseContrast=0.3
        epsilon = np.array([0.2,0])
        numCells = 2

    # detect events
    cellEvents = getCellEvents(spikes, numCells, numExpts, binWidth=binWidth, probThreshold=probThreshold, minProb=minProb)
