'''
visualizations.py

Tools for visualizing data from retinal experiments.

(C) 2014 bnaecker, nirum
'''

import numpy as _np
import matplotlib.pyplot as _plt
import seaborn as _sns
from pyret import filtertools as _ft
from matplotlib import animation as _animation

def raster(spikes, triallength=None, fig=None):
    '''

    Plot a raster of spike times. The `triallength` keyword specifies the
    length of time for each trial, and the `spikes` array is split up into
    segments of that length. These groups are then plotted on top of one
    another, as if they are individual trials.

    Input
    -----

    spikes (ndarray):
        An array of spike times to plot

    triallength (float):
        The length of each trial to stack, in seconds.

    fig (matplotlib figure):
        The figure into which the raster is plotted.

    Output
    ------

    fig (matplotlib figure):
        Matplotlib handle of the figure

    '''

    # Parse time input
    if triallength is None:
        # Compute the time indices of the start and stop of the spikes
        times = _np.array([spikes.min(), spikes.max()])
    else:
        # Compute the time indices of each trial
        times = _np.array([_np.array([0, triallength]) + triallength * i
                for i in _np.arange(_np.ceil(spikes.max() / triallength))])

    # Make a figure
    if not fig or type(fig) is not _plt.Figure:
        fig = _plt.figure()

    # Plot each trial
    ax = fig.add_subplot(111)
    _plt.hold(True)
    for trial in range(times.shape[0]):
        idx = _np.bitwise_and(spikes > times[trial, 0], spikes <= times[trial, 1])
        ax.plot(spikes[idx] - times[trial, 0], (trial + 1) * _np.ones((idx.sum(),1)),
                color='k', linestyle='none', marker='.')

    # Labels etc
    _plt.title('spike raster', fontdict={'fontsize':24})
    _plt.xlabel('time (s)', fontdict={'fontsize':20})
    _plt.ylabel('trial #', fontdict={'fontsize':20})
    _plt.ylim(ymin = 0, ymax=times.shape[0] + 1)
    _plt.show()
    _plt.draw()

    return fig

def psth(spikes, triallength=None, binsize=0.01, fig=None):
    '''

    Plot a PSTH from the given spike times.

    Input
    -----

    spikes (ndarray):
        An array of spike times

    triallength (float):
        The length of each trial to stack, in seconds

    binsize (float):
        The size of bins used in computing the PSTH

    fig (matplotlib figure):
        Figure into which the psth should be plotted

    Output
    ------

    fig (matplotlib figure):
        Matplotlib figure handle

    '''

    # Input-checking
    if not triallength:
        triallength = spikes.max()

    # Compute the histogram bins to use
    ntrials     = int(_np.ceil(spikes.max() / triallength))
    basebins    = _np.arange(0, triallength + binsize, binsize)
    tbins       = _np.tile(basebins, (ntrials, 1)) + (_np.tile(_np.arange(0, ntrials), (basebins.size, 1)).T * triallength)

    # Bin the spikes in each time bin
    bspk = _np.empty((tbins.shape[0], tbins.shape[1] - 1))
    for trial in range(ntrials):
        bspk[trial, :], _ = _np.histogram(spikes, bins=tbins[trial, :])

    # Compute the mean over each trial, and multiply by the binsize
    fr = _np.mean(bspk, axis=0) / binsize

    # Make a figure
    if not fig or type(fig) is not _plt.Figure:
        fig = _plt.figure()

    # Plot the PSTH
    ax = fig.add_subplot(111)
    ax.plot(tbins[0, :-1], fr, color='k', marker=None, linestyle='-', linewidth=2)

    # Labels etc
    _plt.title('psth', fontdict={'fontsize':24})
    _plt.xlabel('time (s)', fontdict={'fontsize':20})
    _plt.ylabel('firing rate (Hz)', fontdict={'fontsize':20})
    _plt.show()
    _plt.draw()

    return fig

def rasterandpsth(spikes, triallength=None, binsize=0.01, fig=None):
    '''

    Plot a spike raster and a PSTH on the same set of axes.

    Input
    -----

    spikes (ndarray):
        An array of spike times

    triallength (float):
        The length of each trial to stack, in seconds

    binsize (float):
        The size of bins used in computing the PSTH

    fig (matplotlib figure):
        Figure into which the psth should be plotted

    Output
    ------

    fig (matplotlib figure):
        Matplotlib figure handle

    '''

    # Input-checking
    if not triallength:
        triallength = spikes.max()

    # Compute the histogram bins to use
    ntrials     = int(_np.ceil(spikes.max() / triallength))
    basebins    = _np.arange(0, triallength + binsize, binsize)
    tbins       = _np.tile(basebins, (ntrials, 1)) + (_np.tile(_np.arange(0, ntrials), (basebins.size, 1)).T * triallength)

    # Bin the spikes in each time bin
    bspk = _np.empty((tbins.shape[0], tbins.shape[1] - 1))
    for trial in range(ntrials):
        bspk[trial, :], _ = _np.histogram(spikes, bins=tbins[trial, :])

    # Compute the mean over each trial, and multiply by the binsize
    fr = _np.mean(bspk, axis=0) / binsize

    # Make a figure
    if not fig or type(fig) is not _plt.Figure:
        fig = _plt.figure()

    # Plot the PSTH
    psthax = fig.add_subplot(111)
    psthax.plot(tbins[0, :-1], fr, color='r', marker=None, linestyle='-', linewidth=2)
    psthax.set_title('psth and raster', fontdict={'fontsize':24})
    psthax.set_xlabel('time (s)', fontdict={'fontsize':20})
    psthax.set_ylabel('firing rate (Hz)', color='r', fontdict={'fontsize':20})
    _sns.set(style='white', context='notebook')
    for tick in psthax.get_yticklabels():
        tick.set_color('r')

    # Plot the raster
    rastax = psthax.twinx()
    _sns.set(style='white', context='notebook')
    _plt.hold(True)
    for trial in range(ntrials):
        idx = _np.bitwise_and(spikes > tbins[trial, 0], spikes <= tbins[trial, -1])
        rastax.plot(spikes[idx] - tbins[trial, 0], trial * _np.ones(spikes[idx].shape), color='k', marker='.', linestyle='none')
    rastax.set_ylabel('trial #', color='k', fontdict={'fontsize':20})
    for tick in psthax.get_yticklabels():
        tick.set_color('k')

    # Show the figure
    _plt.show()
    _plt.draw()

    return fig

def playsta(sta, repeat=True, frametime=100):
    '''

    Plays a spatiotemporal spike-triggered average as a movie

    Input
    -----

    sta (ndarray):
        Spike-triggered average array, shaped as (npix, npix, nframes)

    repeat (boolean) [optional, default=True]:
        Whether or not to repeat the animation

    frametime (float) [optional, default=100]:
        Length of time each frame is displayed for (in milliseconds)

    Output
    ------

    None

    '''

    # Initial frame
    initialFrame = sta[:, :, 0]

    # Set up the figure
    fig     = _plt.figure()
    ax      = _plt.axes(xlim=(0, sta.shape[0]), ylim=(0, sta.shape[1]))
    img     = _plt.imshow(initialFrame)

    # Set up the colors
    maxval = _np.ceil(_np.absolute(sta).max())
    img.set_cmap('gray')
    img.set_interpolation('nearest')

    # Animation function (called sequentially)
    def animate(i):
        ax.set_title('Frame {0:#d}'.format(i + 1))
        img.set_data(sta[:, :, i])

    # Call the animator
    anim = _animation.FuncAnimation(fig, animate, _np.arange(sta.shape[-1]), interval=frametime, repeat=repeat)
    _plt.show()
    _plt.draw()
    return anim

def spatial(spatialFrame, ax=None, clim=None):
    '''

    Plot a spatial filter on a given axes

    Input
    -----

    spatialFrame (ndarray):
        The frame to plot, as an (n x n) matrix.

    ax (matplotlib axes) [optional]:
        the axes on which to plot the data; defaults to creating a new figure

    clim (tuple) [optional]:
        the color range with which to scale the image. Defaults to [-maxval, maxval] where maxval is the max abs. value

    Output
    ------

    ax (matplotlib axes):
        Axes into which the frame is plotted

    '''

    if not ax:
        ax = _plt.figure().add_subplot(111)

    # adjust color limits if necessary
    if not clim:

        # normalize
        spatialFrame -= _np.mean(spatialFrame)

        # find max abs value
        maxabs = _np.max(_np.abs(spatialFrame))

        # set clim
        clim = (-maxabs, maxabs)

    # plot the spatial frame
    img = ax.imshow(spatialFrame, cmap='bwr', interpolation='nearest')
    img.set_clim(clim)
    ax.set_title('Spatial RF')
    ax.set_aspect('equal')

    # add colorbar
    cbar = ax.get_figure().colorbar(img)

    _plt.show()
    _plt.draw()

    return ax

def temporal(time, temporalFilter, ax=None):
    '''

    Plot a temporal filter on a given axes

    Input
    -----

    time (ndarray):
        a time vector to plot against

    temporalFilter (ndarray):
        the temporal filter to plot, has the same dimensions as time

    ax (matplotlib axes) [optional]:
        the axes on which to plot the data; defaults to creating a new figure

    Output
    ------

    ax (matplotlib axes):
        Axes into which the frame is plotted

    '''

    if not ax:
        ax = _plt.figure().add_subplot(111)

    ax.plot(time, temporalFilter, linestyle='-', linewidth=2, color='LightCoral')
    _plt.show()
    _plt.draw()

    return ax

def plotsta1D(sta, ax=None):
    '''

    Plot a 1D spatiotemporal filter (e.g. from a bars experiment)

    '''

    # normalize
    stan = (sta - _np.mean(sta)) / _np.var(sta)
    lim = _np.max(_np.abs(stan))*1.2

    if not ax:
        fig = _plt.figure()
        _sns.set(style='white')
        ax = fig.add_subplot(111)

    im = ax.imshow(stan)
    ax.set_aspect('equal')
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    im.set_clim(-lim,lim)
    im.set_cmap('seismic')
    _plt.show()
    _plt.draw()

def plotsta(time, sta, timeSlice=None):
    '''

    Plot a spatial and temporal filter

    Input
    -----

    time (ndarray):
        a time vector to plot against

    sta (ndarray):
        the filter to plot

    timeslice (int) [optional]:
        the index of the spatial slice to plot

    Output
    ------

    ax (matplotlib axes):
        Axes into which the STA is plotted

    '''

    # create the figure
    fig = _plt.figure()

    # decompose
    spatialProfile, temporalFilter = _ft.decompose(sta)

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

    ell (matplotlibe.pathces.Ellipse object):
        The ellipse to be plotted

    ax (matplotlib axes) [optional]:
        The axes onto which the ellipse should be plotted. Defaults to a new figure

    Output
    ------

    ax  (matplotlib axes):
        The axes onto which the ellipse is plotted

    '''

    # Set some properties
    ell.set_facecolor('green')
    ell.set_alpha(0.5)
    ell.set_edgecolor('black')

    # Create axes or add to given
    if not ax:
        fig = _plt.figure()
        ax = fig.add_subplot(111)
    ax.add_artist(ell)

    _plt.show()
    _plt.draw()
    return ax

def plotcells(cells, ax=None, boxdims=None, start=None, scale=0.25):
    '''

    Plot the receptive fields

    Input
    -----

    cells:
        a list of spatiotemporal receptive fields

    ax (matplotlib axes) [optional]:
        The axes onto which the ellipse should be plotted. Defaults to a new figure

    boxdims [optional, default: None]:
        Dimensions of a box (to indicate the electrode array) to draw behind the cells.
        Should be a tuple containing the (width,height) of the box.

    start [optional, default: None]:
        Location of the lower left corner of the box to draw. If None, the box is centered on the plot.
        Only matters if 'boxdims' is not None.

    Output
    ------

    ax  (matplotlib axes):
        The axes onto which the ellipse is plotted

    '''

    # Create axes or add to given
    if not ax:
        fig = _plt.figure()
        ax = fig.add_subplot(111)

    # define the color palatte
    colors = _sns.color_palette("hls", len(cells))
    _np.random.shuffle(colors)

    # for each cell
    ellipses = list()
    for idx, sta in enumerate(cells):

        # get the spatial profile
        _, _, tidx = _ft.filterpeak(sta)

        # generate ellipse
        ell = _ft.getellipse(sta[:,:,tidx], scale=scale)

        # add it to the plot
        ell.set_facecolor(colors[idx])
        ell.set_edgecolor(colors[idx])
        ell.set_linewidth(2)
        ell.set_linestyle('solid')
        ell.set_alpha(0.3)
        ax.add_artist(ell)
        ellipses.append(ell)

    # add a box to mark the array
    if start is None:
        start = (1-_np.array(boxdims)) / 2.0

    ax.add_patch(_plt.Rectangle((start[0], start[1]), boxdims[0], boxdims[1], fill=False, edgecolor='Black', linestyle='dashed'))
    _plt.xlim(xmin=start[0], xmax=start[0]+boxdims[0])
    _plt.ylim(ymin=start[1], ymax=start[1]+boxdims[1])

    _sns.set_style('nogrid')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    _plt.box('off')
    _plt.tight_layout()
    _plt.show()
    _plt.draw()
    return ax, ellipses

def playrates(rates, patches, palette='gray', numLevels=255, time=None, repeat=True, frametime=100):
    '''
    Plays a movie of the firing rate for N cells by updating the given patches (matplotlib handles)
    (useful in conjunction with the output of plotcells)

    Input
    -----
    rates: an (N, T) matrix of firing rates
    patches: a list of N matplotlib patch elements. The facecolor of these patches is altered according to the rates values
    '''

    # approximate necessary colormap
    colors = _sns.color_palette(palette, numLevels);
    rscale = _np.round( (numLevels-1) * (rates - rates.min()) / (rates.max() - rates.min()) ).astype('int');

    # set up
    fig = _plt.gcf()
    ax = _plt.gca()
    if time is None:
        time = _np.arange(rscale.shape[1])

    # Animation function (called sequentially)
    def animate(t):
        for i in range(rscale.shape[0]):
            patches[i].set_facecolor(colors[rscale[i,t]])
        ax.set_title('Time: %0.2f seconds' % (time[t]), fontsize=20)

    # Call the animator
    anim = _animation.FuncAnimation(fig, animate, _np.arange(rscale.shape[1]), interval=frametime, repeat=repeat)
    _plt.show()
    _plt.draw()
    return anim
