'''
spktools.py

tools for loading and basic manipulation of spike times

(C) 2014 bnaecker, nirum
'''

# Imports
import os, re
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal

def loadspikes(name, date):
    '''
    usage: spk = loadspikes(name, date)
    loads spikes for each cell from the given experiment

    input:
        name	- string experiment name
        date 	- string experiment date
    
    output:
        spk 	- list of spike times for each cell
    '''

    # Path
    datadir = os.path.join(os.path.expanduser(\
                '~/FileCabinet/stanford/baccuslab/projects/'), name, 'data', date)

    # Find all the spike time files
    spkfiles = [os.path.join(datadir, f) \
                for f in os.listdir(datadir) if re.match('c[0-9]{2}.txt', f)]

    # Load the spike times from each file
    return [sp.loadtxt(spkfile) for spkfile in spkfiles]

def binspikes(spk, tmax, binsize=0.01):
    '''
    usage: bspk, tax = binspikes(spk, tmax, binsize)
    bin spike times at the given resolution

    input:
        spk	- list of spike times for each cell
        tmax	- max time (usually end of experiment, last VBL timestamp, etc)
        binsize - size of bins in ms

    output:
        bspk	- list of spike counts in each time bin for each cell
        tax	- the time of the bins
    '''

    # Set up time axis
    tbins = sp.arange(0, tmax, binsize)
    
    # Histogram count for each cell
    bspk = []
    for cell in spk:
        hist, tax = sp.histogram(cell, bins = tbins)
        bspk.append(hist)
    
    return bspk, tax[:-1]

def estfr(bspk, binsize=0.01, npts=7, sd=2):
    '''
    usage: rates = estfr(bspk, npts, sd)
    estimate the instantaneous firing rates from binned spike counts

    input:
        bspk	- list of binned spike counts for each cell (from binspikes)
        npts	- number of points in Gaussian filter used to smooth counts
        sd	- SD (in points) of the Gaussian filter

    output:
        rates	- list of firing rates for each cell
    '''
    # Construct Gaussian filter
    filt = signal.gaussian(npts, sd)

    # Filter each cell's binned spike times
    rates = []
    for cell in bspk:
            rates.append(signal.lfilter(filt, 1, cell) / binsize)
    
    return rates

def raster(spk, cells=None, trange=None):
    '''
    usage: fig = raster(spk, cells, trange)
    plot a raster of spike times over the given time

    input:
        spk	- list of spike times for each cell
        cells	- list of which cells to plot (None == all)
        trange	- 2-elem tuple giving time range (None = (min(spk), max(spk)))

    output:
        fig	- Matplotlib handle of the figure
    '''
    # Input
    if cells is None:
        cells = range(0, len(spk))
    else:
        cells = [c for c in cells if 0 < c <= len(spk)]

    if trange is None:
        trange = (min([s.min() for s in spk]), max([s.max() for s in spk]))
    else:
        trange = (max(trange[0], 0), min(trange[1], max([s.max() for s in spk])))

    # Plot rasters for each cell
    fig = plt.figure()
    ncells = len(cells)
    for cell in range(ncells):
        spikes = spk[cell][sp.logical_and(spk[cell] >= trange[0], spk[cell] < trange[1])]
        plt.plot(spikes, (cell + 1) * sp.ones(spikes.shape), color = 'k', marker = '.', linestyle = 'none')

    # Labels etc
    plt.title('spike rasters', fontdict={'fontsize':24})
    plt.xlabel('time (s)', fontdict={'fontsize':20})
    plt.ylabel('cell #', fontdict={'fontsize':20})
    plt.ylim(ymin = 0, ymax=ncells + 1)
    plt.show()
    
    return fig

def psth(rates, tax, cells=None, trange=None):
    '''
    usage: fig = psth(rates, tax, cells, trange)
    plot psths for the given cells over the given time

    input:
        rates	- list of firing rates for each cell
        tax		- time axis for firing rates
        cells	- list of which cells to plot (None == all)
        trange	- 2-elem tuple giving time range (None == (min(tax), max(tax)))

    output:
        fig	- Matplotlib figure handle
    '''

    # Input
    if cells is None:
            cells = range(0, len(rates))
    else:
            cells = [c for c in cells if 0 < c <= len(rates)]
    ncells = len(cells)

    if trange is None:
            trange = (tax.min(), tax.max())
    else:
            trange = (max(trange[0], 0), min(trange[1], tax.max()))

    # Compute plot indices
    plotinds = sp.logical_and(trange[0] <= tax, tax < trange[1])

    # Compute number of subplots
    n = round(sp.sqrt(ncells))
    nplots = (n, sp.ceil(ncells / n))

    # Plot psths for each cell
    fig = plt.figure()
    for cell in range(ncells):
            plt.subplot(nplots[0], nplots[1], cell + 1)
            plt.plot(tax[plotinds], rates[cell][plotinds], color = 'k', marker = None, linestyle = '-')

            # Labels etc
            plt.title('cell {c} psth'.format(c = cell + 1), fontdict={'fontsize': 24})
            plt.xlabel('time (s)', fontdict={'fontsize':20})
    plt.show()
    
    return fig
