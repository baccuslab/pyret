'''
visualizations.py

Tools for visualizing data from retinal experiments.

(C) 2014 bnaecker, nirum
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import filtertools as ft
from matplotlib import animation

def raster(spikes, triallength=None, fig=None):
    '''
    
    Plot a raster of spike times. The `triallength` keyword specifies the 
    length of time for each trial, and the `spikes` array is split up into 
    segments of that length. These groups are then plotted on top of one 
    another, as if they are individual trials.

    Input
    -----

    spikes:
        An array of spike times to plot

    triallength:
        The length of each trial to stack, in seconds.

    fig:
        The figure into which the raster is plotted.

    Output
    ------

    fig:
        Matplotlib handle of the figure

    '''

    # Parse time input
    if triallength is None:
        # Compute the time indices of the start and stop of the spikes
        times = np.array([spikes.min(), spikes.max()])
    else:
        # Compute the time indices of each trial
        times = np.array([np.array([0, triallength]) + triallength * i
                for i in np.arange(np.ceil(spikes.max() / triallength))])
        
    # Make a figure
    if not fig or type(fig) is not plt.Figure:
        fig = plt.figure()

    # Plot each trial
    ax = fig.add_subplot(111)
    plt.hold(True)
    for trial in range(times.shape[0]):
        idx = np.bitwise_and(spikes > times[trial, 0], spikes <= times[trial, 1])
        ax.plot(spikes[idx] - times[trial, 0], (trial + 1) * np.ones((idx.sum(),1)), 
                color='k', linestyle='none', marker='.')

    # Labels etc
    plt.title('spike raster', fontdict={'fontsize':24})
    plt.xlabel('time (s)', fontdict={'fontsize':20})
    plt.ylabel('trial #', fontdict={'fontsize':20})
    plt.ylim(ymin = 0, ymax=times.shape[0] + 1)
    plt.show()
    plt.draw()

    return fig

def psth(spikes, triallength=None, binsize=0.01, fig=None):
    '''
    
    Plot a PSTH from the given spike times.

    Input
    -----

    spikes:
        An array of spike times

    triallength:
        The length of each trial to stack, in seconds

    binsize:
        The size of bins used in computing the PSTH

    fig:
        Figure into which the psth should be plotted

    Output
    ------

    fig:
        Matplotlib figure handle

    '''

    # Input-checking
    if not triallength:
        triallength = spikes.max()

    # Compute the histogram bins to use
    ntrials = int(np.ceil(spikes.max() / triallength))
    basebins = np.arange(0, triallength + binsize, binsize)
    tbins = np.tile(basebins, (ntrials, 1)) + (np.tile(np.arange(0, ntrials), (basebins.size, 1)).T * triallength)
        
    # Bin the spikes in each time bin
    bspk = np.empty((tbins.shape[0], tbins.shape[1] - 1))
    for trial in range(ntrials):
        bspk[trial, :], _ = np.histogram(spikes, bins=tbins[trial, :])

    # Compute the mean over each trial, and multiply by the binsize
    fr = np.mean(bspk, axis=0) / binsize

    # Make a figure
    if not fig or type(fig) is not plt.Figure:
        fig = plt.figure()

    # Plot the PSTH
    ax = fig.add_subplot(111)
    ax.plot(tbins[0, :-1], fr, color='k', marker=None, linestyle='-', linewidth=2)

    # Labels etc
    plt.title('psth', fontdict={'fontsize':24})
    plt.xlabel('time (s)', fontdict={'fontsize':20})
    plt.ylabel('firing rate (Hz)', fontdict={'fontsize':20})
    plt.show()
    plt.draw()

    return fig

def rasterandpsth(spikes, triallength=None, binsize=0.01, fig=None):
    '''

    Plot a spike raster and a PSTH on the same set of axes.

    Input
    -----

    spikes:
        An array of spike times

    triallength:
        The length of each trial to stack, in seconds

    binsize:
        The size of bins used in computing the PSTH

    fig:
        Figure into which the psth should be plotted

    Output
    ------

    fig:
        Matplotlib figure handle

    '''

    # Input-checking
    if not triallength:
        triallength = spikes.max()

    # Compute the histogram bins to use
    ntrials = int(np.ceil(spikes.max() / triallength))
    basebins = np.arange(0, triallength + binsize, binsize)
    tbins = np.tile(basebins, (ntrials, 1)) + (np.tile(np.arange(0, ntrials), (basebins.size, 1)).T * triallength)

    # Bin the spikes in each time bin
    bspk = np.empty((tbins.shape[0], tbins.shape[1] - 1))
    for trial in range(ntrials):
        bspk[trial, :], _ = np.histogram(spikes, bins=tbins[trial, :])

    # Compute the mean over each trial, and multiply by the binsize
    fr = np.mean(bspk, axis=0) / binsize

    # Make a figure
    if not fig or type(fig) is not plt.Figure:
        fig = plt.figure()

    # Plot the PSTH
    psthax = fig.add_subplot(111)
    psthax.plot(tbins[0, :-1], fr, color='r', marker=None, linestyle='-', linewidth=2)
    psthax.set_title('psth and raster', fontdict={'fontsize':24})
    psthax.set_xlabel('time (s)', fontdict={'fontsize':20})
    psthax.set_ylabel('firing rate (Hz)', color='r', fontdict={'fontsize':20})
    sns.set_axes_style('nogrid', 'notebook')
    for tick in psthax.get_yticklabels():
        tick.set_color('r')

    # Plot the raster
    rastax = psthax.twinx()
    sns.set_axes_style('nogrid', 'notebook')
    plt.hold(True)
    for trial in range(ntrials):
        idx = np.bitwise_and(spikes > tbins[trial, 0], spikes <= tbins[trial, -1])
        rastax.plot(spikes[idx] - tbins[trial, 0], trial * np.ones(spikes[idx].shape), color='k', marker='.', linestyle='none')
    rastax.set_ylabel('trial #', color='k', fontdict={'fontsize':20})
    for tick in psthax.get_yticklabels():
        tick.set_color('k')

    # Show the figure
    plt.show()
    plt.draw()

    return fig

def playsta(sta, repeat=True, frametime=100):
    '''
    
    Plays a spatiotemporal spike-triggered average as a movie

    Input
    -----

    sta:
        Spike-triggered average array, shaped as (npix, npix, nframes)

    repeat [optional, default=True]:
        Whether or not to repeat the animation

    frametime [optional, default=100]:
        Length of time each frame is displayed for (in milliseconds)

    Output
    ------

    None

    '''

    # Initial frame
    initialFrame = sta[:, :, 0]

    # Set up the figure
    fig = plt.figure()
    ax = plt.axes(xlim=(0, sta.shape[0]), ylim=(0, sta.shape[1]))
    img = plt.imshow(initialFrame)

    # Set up the colors
    maxval = np.ceil(np.absolute(sta).max())
    img.set_cmap('gray')
    img.set_interpolation('nearest')
    plt.colorbar()

    # Animation initialization function
    def init():
        img.set_data(initialFrame)
        return img

    # Animation function (called sequentially)
    def animate(i):
        ax.set_title('Frame {0:#d}'.format(i + 1))
        img.set_data(sta[:, :, i])
        return img

    # Call the animator
    anim = animation.FuncAnimation(fig, animate,
                np.arange(sta.shape[-1]), init_func=init, interval=frametime, repeat=False)
    plt.show()
    plt.draw()

def spatial(spatialFrame, ax=None):
    '''
	
    Plot a spatial filter on a given axes

    Input
    -----

    spatialFrame:
        The frame to plot, as an (n x n) matrix.

    ax [optional]:
        the axes on which to plot the data; defaults to creating a new figure

    Output
    ------

    axes handle

    '''

    if not ax:
        ax = plt.figure().add_subplot(111)

    img = ax.imshow(spatialFrame)
    img.set_cmap('RdBu')
    img.set_interpolation('nearest')
    plt.colorbar()
    plt.show()
    plt.draw()

    return ax

def temporal(time, temporalFilter, ax=None):
    '''
	
    Plot a temporal filter on a given axes

    Input
    -----

    time:
        a time vector to plot against

    temporalFilter:
        the temporal filter to plot, has the same dimensions as time

    ax [optional]:
        the axes on which to plot the data; defaults to creating a new figure

    Output
    ------

    axes handle

    '''

    if not ax:
        ax = plt.figure().add_subplot(111)

    ax.plot(time, temporalFilter, linestyle='-', linewidth=2, color='LightCoral')
    plt.show()
    plt.draw()

    return ax

def plotsta(time, sta, timeSlice=None):
    '''
	
    Plot a spatial and temporal filter

    Input
    -----

    time:
        a time vector to plot against

    sta:
        the filter to plot

    timeslice [optional]:
        the index of the spatial slice to plot

    Output
    ------

    axes handle

    '''

    # create the figure
    fig = plt.figure()

    # decompose
    spatialProfile, temporalFilter = ft.decompose(sta)

    # plot spatial profile
    axspatial = spatial(spatialProfile, fig.add_subplot(121))

    # plot temporal profile
    axtemporal = temporal(time, temporalFilter, fig.add_subplot(122))

    # return handles
    return fig, (axspatial, axtemporal)

def ellipse(ell, ax=None):
    '''
    
    Plot the given ellipse, fit to the spatial receptive field of a cell

    Input
    -----

    ell:
        A matplotlib.patches.Ellipse object

    ax [optional]:
        The axes onto which the ellipse should be plotted. Defaults to a new figure

    Output
    ------

    ax:
        The axes onto which the ellipse is plotted
    
    '''

    # Set some properties
    ell.set_facecolor('green')
    ell.set_alpha(0.5)
    ell.set_edgecolor('black')

    # Create axes or add to given
    if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
    ax.add_artist(ell)

    plt.show()
    plt.draw()
    return ax
